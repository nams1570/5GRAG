from typing import Callable
from utils import Document, getTokenCount
from MetadataAwareChunker import getSectionedChunks,addExtraDocumentWideMetadataForReason, getFullSectionChunks
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from settings import config
import os 
import time
class DBClient:
    def getDocsFromFilePath(self,file_list:list[str],metadata_func:Callable[[str,str],dict]=addExtraDocumentWideMetadataForReason,doc_dir:str=config["DOC_DIR"],useFullSectionChunks:bool=False)->list[Document]:
        """@file_list: list(str) of file names. Not absolute/relative paths
        @metadata_func: function that will be used to extract the document wide metadata
        @doc_dir: directory where the documents are located
        """
        docs = []
        file_list = [os.path.join(doc_dir,file) for file in file_list]
        if useFullSectionChunks:
            docs = getFullSectionChunks(file_list,addExtraDocumentWideMetadata=metadata_func)
        else:
            docs = getSectionedChunks(file_list,addExtraDocumentWideMetadata=metadata_func)
        return docs
    
    def _add_doc_list_to_db(self,docs:list[Document]):
        pass
    
    def _safe_add_docs(self,docs,batch_num,attempt=1,max_attempts=3):
        total_tokens = sum(getTokenCount(d.page_content,model_name=self.embedding_model_name) for d in docs)
        MAX_TOKENS_ALLOWED = 270000

        if total_tokens > MAX_TOKENS_ALLOWED:
            print(f"Batch {batch_num} is too large")
            mid = len(docs) //2
            self._safe_add_docs(docs[:mid],batch_num=f"{batch_num}a")
            self._safe_add_docs(docs[mid:],batch_num=f"{batch_num}b")
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
    
    def _get_embedding_model_function(self,embedding_model_name:str):
        """Returns the embedding function for the given model name. Currently only supports OpenAI models."""
        return OpenAIEmbeddingFunction(model_name=embedding_model_name)
    
    def __init__(self,embedding_model_name:str="text-embedding-3-large",collection_name:str=config["SPEC_COLL_NAME"],db_dir_path:str=config["CHROMA_DIR"]):
        #construct chroma base db     
        self.chroma_client = chromadb.PersistentClient(path=db_dir_path)
        self.embedding_model_name = embedding_model_name
        embeddings = self._get_embedding_model_function(embedding_model_name)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name, embedding_function=embeddings)

    def updateDBFromFileList(self,new_file_list:list[str],metadata_func:Callable[[str,str],dict]=addExtraDocumentWideMetadataForReason,doc_dir:str=config["DOC_DIR"],useFullSectionChunks:bool=False):
        """@new_file_list: list(str) list of file names (not abs paths)
        Turn the new files in DOC_DIR into a list of documents and add them
        to the vector store."""
        new_docs = self.getDocsFromFilePath(new_file_list,metadata_func=metadata_func,doc_dir=doc_dir,useFullSectionChunks=useFullSectionChunks)
        self.add_docs_to_db(new_docs)

    def delFromDB(self):
        """This is a placeholder function. 
        Whilst doing this definitely prevents the retriever from fetching things with these docs as source,
        It still seems able to answer questions based on it."""
        self.vector_db.delete(where={'source':{'$eq':'data/38214-hc0.docx'}})
        self.vector_db.delete(where={'source':{'$eq':'data/38176-2-gc0.docx'}})
        self.vector_db.delete(where={'source':{'$eq':'data/38741-i31.docx'}})

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