from DBClient import DBClient
from MetadataAwareChunker import getSectionedChunks,addExtraDocumentWideMetadataForReason
from settings import config
import os
from langchain_openai import OpenAIEmbeddings
from utils import convertAllDocToDocx,getAllFilesInDirMatchingFormat

if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large',api_key=config["API_KEY"])
    db = DBClient(embedding_model=embeddings,collection_name=config["TDOC_COLL_NAME"],db_dir_path=config["CHROMA_DIR"],doc_dir_path="reasoning")

    convertAllDocToDocx("reasoning")
    file_list = getAllFilesInDirMatchingFormat("reasoning")

    docs = []
    for file in file_list:
        docs += getSectionedChunks([os.path.join("reasoning",file)],addExtraDocumentWideMetadataForReason)

    db.vector_db.add_documents(docs)
    retriever = db.getRetriever()
    