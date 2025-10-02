from ast import parse
from DBClient import DBClient
from settings import config
import os
import time
from langchain_openai import OpenAIEmbeddings
from utils import convertAllDocToDocx,getAllFilesInDirMatchingFormat, getTokenCount
from MetadataAwareChunker import getSectionedChunks

COLLECTION_NAME = ""
DB_DIR_PATH = os.path.join("baseline","db")
DOC_DIR_PATH = os.path.join("./all3gppdocsfromrel17and18","Release 17")

def _safe_add_docs(docs,db,batch_num,attempt=1,max_attempts=3):
        total_tokens = sum(getTokenCount(d.page_content,model_name="text-embedding-3-large") for d in docs)
        MAX_TOKENS_ALLOWED = 270000

        if total_tokens > MAX_TOKENS_ALLOWED:
            print(f"Batch {batch_num} is too large")
            mid = len(docs) //2
            _safe_add_docs(docs[:mid],db=db,batch_num=f"{batch_num}a")
            _safe_add_docs(docs[mid:],db=db,batch_num=f"{batch_num}b")
            return
        try:
            db.vector_db.add_documents(docs)
            print(f"just added batch {batch_num}")
        except Exception as e:
            if attempt < max_attempts:
                wait = attempt * 30
                print(f"Error ob batch {batch_num}, attempt {attempt} {e}. Retrying in {wait}s...")
                time.sleep(wait)
                _safe_add_docs(docs,db=db,batch_num=batch_num,attempt=attempt+1,max_attempts=max_attempts)
            else:
                print(f"Failed batch {batch_num} after {max_attempts} attempts.")
                raise e
            


def parse_spec_list(file_list,db):
    docs = getSectionedChunks([os.path.join(DOC_DIR_PATH,file) for file in file_list])
    
    batch_size = 1000
    i=1
    while (i-1) * batch_size < len(docs):
        print(f" ****** \n\n number of chunks is {len(docs)} and we are on batch {i}. \n\n")
        _safe_add_docs(docs[(i-1)*batch_size:i*batch_size],db=db,batch_num=i,max_attempts=3)
        i+=1
    print(f"Added {len(docs)} chunks to the database")


if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large',api_key=config["API_KEY"])
    db = DBClient(embedding_model=embeddings,collection_name=COLLECTION_NAME,db_dir_path=DB_DIR_PATH,doc_dir_path=DOC_DIR_PATH)

    file_list = getAllFilesInDirMatchingFormat(DOC_DIR_PATH)
    parse_spec_list(file_list,db)


