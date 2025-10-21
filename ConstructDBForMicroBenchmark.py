from settings import config
from DBClient import DBClient
from utils import convertAllDocToDocx,getAllFilesInDirMatchingFormat, getTokenCount
from MetadataAwareChunker import addExtraDocumentWideMetadataForReason
from CollectionNames import CROSS_CONTEXT_BENCHMARK_COLL_NAME

COLLECTION_NAME = CROSS_CONTEXT_BENCHMARK_COLL_NAME
DB_DIR_PATH = "fakedb"
DOC_DIR_PATH = "dataformicrobenchmark"

def parse_docs(file_list,db:DBClient):
    db.updateDBFromFileList(file_list,metadata_func=addExtraDocumentWideMetadataForReason,doc_dir=DOC_DIR_PATH,useFullSectionChunks=True)

if __name__ == "__main__":
    db = DBClient(embedding_model_name='text-embedding-3-large',collection_name=COLLECTION_NAME,db_dir_path=DB_DIR_PATH)

    file_list = getAllFilesInDirMatchingFormat(DOC_DIR_PATH)
    parse_docs(file_list,db)
    print(db.queryDB("What is 5G?",k=3))