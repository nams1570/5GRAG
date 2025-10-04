from pydoc import doc
from webbrowser import get
from ChangeTracker import ChangeTracker, get_version_preceding_first_in_release, get_doc_list_for_version_preceding_first
from DBClient import DBClient
from MetadataAwareChunker import getFullSectionChunks,addExtraDocumentWideMetadataForReason
from settings import config
import os
from langchain_openai import OpenAIEmbeddings
from utils import getAllFilesInDirMatchingFormat,convertAllDocToDocx, getTokenCount
from CollectionNames import DIFFS as DIFF_COLL_NAME
import time

DIFF_DOC_DIR = "./all3gppdocsfromrel17and18/docsfordiffs"
DIFF_DB_DIR_PATH = "baseline/db"


def get_sorted_versions(versionMap:dict)->list[str]:
    versions = sorted(versionMap.keys(), key=lambda v: list(map(int, v.split("."))))
    return versions

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


if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large',api_key=config["API_KEY"])
    db = DBClient(embedding_model=embeddings,collection_name=DIFF_COLL_NAME,db_dir_path=DIFF_DB_DIR_PATH,doc_dir_path=DIFF_DOC_DIR)
        
    #convertAllDocToDocx(DIFF_DOC_DIR)
    file_list = getAllFilesInDirMatchingFormat(DIFF_DOC_DIR)
    print(file_list)

    docIDToversionToChunks = {}

    all_chunks = getFullSectionChunks([os.path.join(DIFF_DOC_DIR,file) for file in file_list],addExtraDocumentWideMetadataForReason)

    for chunk in all_chunks:
        docID = chunk.metadata["docID"]
        if docID not in docIDToversionToChunks:
            docIDToversionToChunks[docID] = {}

        fileVersion = chunk.metadata["version"]
        x,y,z = map(int,fileVersion.split("."))
        releaseNum = x 
        VERSION_PRECEDING_FIRST_IN_RELEASE = get_version_preceding_first_in_release(x)
        if y == 0 and z == 0:
            docIDToversionToChunks[docID][VERSION_PRECEDING_FIRST_IN_RELEASE] = get_doc_list_for_version_preceding_first(chunk.metadata,VERSION_PRECEDING_FIRST_IN_RELEASE)
        docIDToversionToChunks[docID][fileVersion] = docIDToversionToChunks[docID].get(fileVersion,[]) + [chunk]
    
    docs = []

    for docId, versionMap in docIDToversionToChunks.items():
        versions = get_sorted_versions(versionMap)

        for i in range(len(versions) - 1):
            fromVersion = versions[i]
            toVersion = versions[i+1]

            if ChangeTracker.areAdjacentVersions(fromVersion, toVersion):
                docs.extend(
                    ChangeTracker.createDBDocumentsForAdjacentVersions(
                        versionMap[fromVersion],
                        versionMap[toVersion]
                    )
                )


    batch_size = 1000
    i=1
    while (i-1) * batch_size < len(docs):
        print(f" ****** \n\n number of chunks is {len(docs)} and we are on batch {i}. \n\n")
        _safe_add_docs(docs[(i-1)*batch_size:i*batch_size],vector_store=db.vector_db,batch_num=i,max_attempts=3)
        i+=1
    print(f"Added {len(docs)} chunks to the database")

    retriever = db.getRetriever()
    print(retriever.invoke("Random access preambles can only be transmitted"))