from typing import Callable
import uuid
from utils import Document, getTokenCount, deterministic_id, RequestedChunkingType
from MetadataAwareChunker import getSectionedChunks,addExtraDocumentWideMetadataForReason, getFullSectionChunks, getCRChunks
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from settings import config
import os 
import time

class DBClient:
    def getDocsFromFilePath(self,file_list:list[str],metadata_func:Callable[[str,str],dict]=addExtraDocumentWideMetadataForReason,doc_dir:str=config["DOC_DIR"],requested_chunking_type: RequestedChunkingType=RequestedChunkingType.SECTION)->list[Document]:
        """@file_list: list(str) of file names. Not absolute/relative paths
        @metadata_func: function that will be used to extract the document wide metadata
        @doc_dir: directory where the documents are located
        """
        docs = []
        file_list = [os.path.join(doc_dir,file) for file in file_list]
        match requested_chunking_type:
            case RequestedChunkingType.FULL_SECTION:
                docs = getFullSectionChunks(file_list,addExtraDocumentWideMetadata=metadata_func)
            case RequestedChunkingType.SECTION:
                docs = getSectionedChunks(file_list,addExtraDocumentWideMetadata=metadata_func)
            case RequestedChunkingType.CR:
                docs = getCRChunks(file_list)
            case _:
                raise ValueError(f"Unknown requested_chunking_type: {requested_chunking_type}")
        return docs

    def _add_doc_list_to_db(self,docs:list[Document]):
        """Adds the given list of documents to the collection"""
        if len(docs) ==0:
            print("No documents to add to DB")
            return
        document_texts = []
        uuids = []
        metadatas = []
        for doc in docs:
            if not doc.page_content or not doc.metadata or doc.page_content.strip() == "" or doc.metadata =={}:
                print(f"Skipping doc with empty content or metadata: {doc}")
                continue
            uuids.append(deterministic_id(doc.page_content, doc.metadata))
            document_texts.append(doc.page_content)
            metadatas.append(doc.metadata)

        if len(document_texts) != len(metadatas)  or len(uuids) != len(metadatas) or len(uuids) != len(document_texts):
            raise ValueError("DBClient: documents and metadatas length mismatch before add")

        try:
            self.collection.add(documents=document_texts,metadatas=metadatas,ids=uuids)
        except Exception as e:
            print(f"Error adding documents to DB: {e}")
            uuids = [str(uuid.uuid4()) for _ in range(len(document_texts))]
            self.collection.add(documents=document_texts,metadatas=metadatas,ids=uuids)

    def _safe_add_docs(self,docs:list[Document],batch_num:int|str,attempt:int=1,max_attempts:int=3):
        if not docs or len(docs) == 0:
            print(f"Batch {batch_num} is empty — skipping.")
            return

        total_tokens = sum(getTokenCount(d.page_content,model_name=self.embedding_model_name) for d in docs)
        MAX_TOKENS_ALLOWED = 8191

        if total_tokens > MAX_TOKENS_ALLOWED and len(docs) > 1:
            print(f"Batch {batch_num} is too large")
            mid = len(docs) //2
            self._safe_add_docs(docs[:mid],batch_num=f"{batch_num}a")
            self._safe_add_docs(docs[mid:],batch_num=f"{batch_num}b")
            return
        elif total_tokens > MAX_TOKENS_ALLOWED:
            print(f"\n\n **Batch {batch_num} has a single document that is too large to add to the DB. Skipping document.")
            return
        try:
            self._add_doc_list_to_db(docs)
            print(f"just added batch {batch_num}")
        except Exception as e:
            if attempt < max_attempts:
                wait = attempt * 30
                print(f"Error on batch {batch_num}, attempt {attempt} {e}. Retrying in {wait}s...")
                time.sleep(wait)
                self._safe_add_docs(docs,batch_num=batch_num,attempt=attempt+1,max_attempts=max_attempts)
            else:
                print(f"Failed batch {batch_num} after {max_attempts} attempts.")
                raise e
    
    def add_docs_to_db(self,docs:list[Document]):
        batch_size = 1000
        i=1
        while (i-1) * batch_size < len(docs):
            print(f" ****** \n\n number of chunks is {len(docs)} and we are on batch {i}. \n\n")
            self._safe_add_docs(docs[(i-1)*batch_size:i*batch_size],batch_num=i,max_attempts=3)
            i+=1
        print(f"added {len(docs)} documents to the database")

    def _get_embedding_model_function(self,embedding_model_name:str):
        """Returns the embedding function for the given model name. Currently only supports OpenAI models."""
        return OpenAIEmbeddingFunction(model_name=embedding_model_name,api_key=config["API_KEY"])
    
    def __init__(self,embedding_model_name:str="text-embedding-3-large",collection_name:str=config["SPEC_COLL_NAME"],db_dir_path:str=config["CHROMA_DIR"]):
        #construct chroma base db     
        self.chroma_client = chromadb.PersistentClient(path=db_dir_path)
        self.embedding_model_name = embedding_model_name
        embeddings = self._get_embedding_model_function(embedding_model_name)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name, embedding_function=embeddings)

    def updateDBFromFileList(self,new_file_list:list[str],metadata_func:Callable[[str,str],dict]=addExtraDocumentWideMetadataForReason,doc_dir:str=config["DOC_DIR"],requested_chunking_type: RequestedChunkingType=RequestedChunkingType.SECTION):
        """@new_file_list: list(str) list of file names (not abs paths)
        Turn the new files in DOC_DIR into a list of documents and add them
        to the vector store."""
        new_docs = self.getDocsFromFilePath(new_file_list,metadata_func=metadata_func,doc_dir=doc_dir,requested_chunking_type=requested_chunking_type)
        self.add_docs_to_db(new_docs)

    def delFromDB(self,filter):
        """Delete all documents from the DB that match the given filter.
        @filter: metadata filter to apply to the deletion. Follow chroma syntax for filtering at https://docs.trychroma.com/docs/querying-collections/metadata-filtering"""
        self.collection.delete(where=filter)

    def queryDB(self,query_text:str,k:int,filter:dict={})->list[Document]:
        """k is how many docs to retrieve, query_text is what we query with.
        @filter: metadata filter to apply to the query. Follow chroma syntax for filtering at https://docs.trychroma.com/docs/querying-collections/metadata-filtering"""
        if filter == {}:
            db_resp = self.collection.query(query_texts=[query_text],n_results=k)
        else:
            db_resp = self.collection.query(query_texts=[query_text],n_results=k,where=filter)

        docs = []
        for doc,meta in zip(db_resp['documents'][0],db_resp['metadatas'][0]):
            docs.append(Document(page_content=doc,metadata=meta))
        return docs

    
    
if __name__ == "__main__":
    file_list = ["38214-hc0.docx"]
    print(DBClient.addDocsFromFilePath(file_list))