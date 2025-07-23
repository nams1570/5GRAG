from ChangeTracker import ChangeTracker
from DBClient import DBClient
from MetadataAwareChunker import getFullSectionChunks
from settings import config
import os
from langchain_openai import OpenAIEmbeddings


if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large',api_key=config["API_KEY"])
    db = DBClient(embedding_model=embeddings,collection_name="diff",db_dir_path=config["CHROMA_DIR"],doc_dir_path="testchange")
    file_list = ["38211-i60.docx","38211-i40.docx","38211-i50.docx","38211-i70.docx"]


    versionToChunks = {}
    for file in file_list:
        chunks = getFullSectionChunks([os.path.join("testchange",file)])
        versionToChunks[chunks[0].metadata["version"]] = chunks
    
    docs = []

    for fromVersion in versionToChunks.keys():
        for toVersion in versionToChunks.keys():
            if ChangeTracker.areAdjacentVersions(fromVersion=fromVersion,toVersion=toVersion):
                docs.extend(ChangeTracker.createDBDocumentsForAdjacentVersions(versionToChunks[fromVersion],versionToChunks[toVersion]))

    db.vector_db.add_documents(docs)

    retriever = db.getRetriever()
    print(retriever.invoke("Random access preambles can only be transmitted"))