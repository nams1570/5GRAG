from DBClient import DBClient
from MetadataAwareChunker import getSectionedChunks,addExtraDocumentWideMetadataForReason, getCRChunks
from settings import config
import os
from langchain_openai import OpenAIEmbeddings
from utils import convertAllDocToDocx,getAllFilesInDirMatchingFormat, getTokenCount
from CollectionNames import REASONING_DOCS
import time

COLLECTION_NAME =  REASONING_DOCS
DB_DIR_PATH = os.path.join("baseline","db")
DOC_DIR_PATH = os.path.join("./all3gppdocsfromrel17and18","CRs")


def _safe_add_docs(docs,batch_num,vector_store,attempt=1,max_attempts=3):
    total_tokens = sum(getTokenCount(d.page_content,model_name="text-embedding-3-large") for d in docs)
    MAX_TOKENS_ALLOWED = 270000

    if total_tokens > MAX_TOKENS_ALLOWED:
        print(f"Batch {batch_num} is too large")
        mid = len(docs) //2
        _safe_add_docs(docs[:mid],batch_num=f"{batch_num}a",vector_store=vector_store,attempt=attempt,max_attempts=max_attempts)
        _safe_add_docs(docs[mid:],batch_num=f"{batch_num}b",vector_store=vector_store)
        return
    try:
        vector_store.add_documents(docs)
        print(f"just added batch {batch_num}")
    except Exception as e:
        if attempt < max_attempts:
            wait = attempt * 30
            print(f"Error ob batch {batch_num}, attempt {attempt} {e}. Retrying in {wait}s...")
            time.sleep(wait)
            _safe_add_docs(docs,batch_num,vector_store=vector_store,attempt=attempt+1,max_attempts=max_attempts)
        else:
            print(f"Failed batch {batch_num} after {max_attempts} attempts.")
            raise e

def parse_nonCR_docs(file_list,db):
    docs = []
    for file in file_list:
        docs += getSectionedChunks([os.path.join(DOC_DIR_PATH,file)],addExtraDocumentWideMetadataForReason)
    
    db.vector_db.add_documents(docs)

def parse_CR_docs(file_list,db):
    docs = getCRChunks([os.path.join(DOC_DIR_PATH,file) for file in file_list])
    print(f"Number of CR chunks is {len(docs)}, now adding to DB")

    batch_size = 1000
    i=1
    while (i-1) * batch_size < len(docs):
        print(f" ****** \n\n number of chunks is {len(docs)} and we are on batch {i}. \n\n")
        _safe_add_docs(docs[(i-1)*batch_size:i*batch_size],vector_store=db.vector_db,batch_num=i,max_attempts=3)
        i+=1
    print(f"Added {len(docs)} chunks to the database")

if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large',api_key=config["API_KEY"])
    db = DBClient(embedding_model=embeddings,collection_name=COLLECTION_NAME,db_dir_path=DB_DIR_PATH,doc_dir_path=DOC_DIR_PATH)

    #convertAllDocToDocx(DOC_DIR_PATH)
    file_list = getAllFilesInDirMatchingFormat(DOC_DIR_PATH)
    print(file_list)
    #parse_nonCR_docs(file_list,db)
    parse_CR_docs(file_list,db)

    