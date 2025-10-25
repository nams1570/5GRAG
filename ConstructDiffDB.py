from ChangeTracker import ChangeTracker, get_version_preceding_first_in_release, get_doc_list_for_version_preceding_first
from DBClient import DBClient
from MetadataAwareChunker import getFullSectionChunks,addExtraDocumentWideMetadataForReason
import os
from utils import getAllFilesInDirMatchingFormat,convertAllDocToDocx, getTokenCount, RequestedChunkingType
from CollectionNames import DIFFS as DIFF_COLL_NAME

DIFF_DOC_DIR = "./all3gppdocsfromrel17and18/docsfordiffs"
DIFF_DB_DIR_PATH = "fakedb"


def get_sorted_versions(versionMap:dict)->list[str]:
    versions = sorted(versionMap.keys(), key=lambda v: list(map(int, v.split("."))))
    return versions

if __name__ == "__main__":
    db = DBClient(embedding_model_name="text-embedding-3-large",collection_name=DIFF_COLL_NAME,db_dir_path=DIFF_DB_DIR_PATH)

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

    db.add_docs_to_db(docs)
    print(f"Added {len(docs)} chunks to the database")

    print(db.queryDB("Random access preambles can only be transmitted"))