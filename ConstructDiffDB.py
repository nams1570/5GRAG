from ChangeTracker import ChangeTracker, get_version_preceding_first_in_release, get_doc_list_for_version_preceding_first
from DBClient import DBClient
from MetadataAwareChunker import getFullSectionChunks,addExtraDocumentWideMetadataForReason
from settings import config
import os
from langchain_openai import OpenAIEmbeddings
from utils import getAllFilesInDirMatchingFormat,convertAllDocToDocx

DIFF_DOC_DIR = "testchange"

if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large',api_key=config["API_KEY"])
    db = DBClient(embedding_model=embeddings,collection_name=config["DIFF_COLL_NAME"],db_dir_path=config["CHROMA_DIR"],doc_dir_path=DIFF_DOC_DIR)
        
    convertAllDocToDocx(DIFF_DOC_DIR)
    file_list = getAllFilesInDirMatchingFormat(DIFF_DOC_DIR)
    print(file_list)

    versionToChunks = {}
    for file in file_list:
        chunks = getFullSectionChunks([os.path.join(DIFF_DOC_DIR,file)],addExtraDocumentWideMetadataForReason)

        fileVersion = chunks[0].metadata["version"]
        x,y,z = map(int,chunks[0].metadata["version"].split("."))
        releaseNum = x 
        VERSION_PRECEDING_FIRST_IN_RELEASE = get_version_preceding_first_in_release(x)
        if y == 0 and z == 0:
            versionToChunks[VERSION_PRECEDING_FIRST_IN_RELEASE] = get_doc_list_for_version_preceding_first(chunks[0].metadata,VERSION_PRECEDING_FIRST_IN_RELEASE)
        versionToChunks[chunks[0].metadata["version"]] = chunks
    
    docs = []

    for fromVersion in versionToChunks.keys():
        for toVersion in versionToChunks.keys():
            if ChangeTracker.areAdjacentVersions(fromVersion=fromVersion,toVersion=toVersion):
                docs.extend(ChangeTracker.createDBDocumentsForAdjacentVersions(versionToChunks[fromVersion],versionToChunks[toVersion]))

    db.vector_db.add_documents(docs)

    retriever = db.getRetriever()
    print(retriever.invoke("Random access preambles can only be transmitted"))