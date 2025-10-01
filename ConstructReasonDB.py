from DBClient import DBClient
from MetadataAwareChunker import getSectionedChunks,addExtraDocumentWideMetadataForReason, getCRChunks
from controller import DB_DIR
from settings import config
import os
from langchain_openai import OpenAIEmbeddings
from utils import convertAllDocToDocx,getAllFilesInDirMatchingFormat

def parse_nonCR_docs(file_list,db):
    docs = []
    for file in file_list:
        docs += getSectionedChunks([os.path.join("reasoning",file)],addExtraDocumentWideMetadataForReason)
    
    db.vector_db.add_documents(docs)

def parse_CR_docs(file_list,db):
    docs = getCRChunks([os.path.join("reasoning",file) for file in file_list])
    db.vector_db.add_documents(docs)

COLLECTION_NAME = config["TDOC_COLL_NAME"]
DB_DIR_PATH = config["CHROMA_DIR"]
DOC_DIR_PATH = "reasoning"

if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large',api_key=config["API_KEY"])
    db = DBClient(embedding_model=embeddings,collection_name=COLLECTION_NAME,db_dir_path=DB_DIR_PATH,doc_dir_path=DOC_DIR_PATH)

    convertAllDocToDocx(DOC_DIR_PATH)
    file_list = getAllFilesInDirMatchingFormat(DOC_DIR_PATH)

    #parse_nonCR_docs(file_list,db)

    