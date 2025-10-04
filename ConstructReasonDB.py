from DBClient import DBClient
from MetadataAwareChunker import getSectionedChunks,addExtraDocumentWideMetadataForReason, getCRChunks
from controller import DB_DIR
from graph.GraphNode import RE
from settings import config
import os
from langchain_openai import OpenAIEmbeddings
from utils import convertAllDocToDocx,getAllFilesInDirMatchingFormat
from CollectionNames import REASONING_DOCS

COLLECTION_NAME =  REASONING_DOCS
DB_DIR_PATH = os.path.join("baseline","db")
DOC_DIR_PATH = os.path.join("./all3gppdocsfromrel17and18","CRs")

def parse_nonCR_docs(file_list,db):
    docs = []
    for file in file_list:
        docs += getSectionedChunks([os.path.join(DOC_DIR_PATH,file)],addExtraDocumentWideMetadataForReason)
    
    db.vector_db.add_documents(docs)

def parse_CR_docs(file_list,db):
    docs = getCRChunks([os.path.join(DOC_DIR_PATH,file) for file in file_list])
    db.vector_db.add_documents(docs)

if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large',api_key=config["API_KEY"])
    db = DBClient(embedding_model=embeddings,collection_name=COLLECTION_NAME,db_dir_path=DB_DIR_PATH,doc_dir_path=DOC_DIR_PATH)

    #convertAllDocToDocx(DOC_DIR_PATH)
    file_list = getAllFilesInDirMatchingFormat(DOC_DIR_PATH)
    print(file_list)
    #parse_nonCR_docs(file_list,db)
    parse_CR_docs(file_list,db)

    