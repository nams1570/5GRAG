from DBClient import DBClient
from MetadataAwareChunker import addExtraDocumentWideMetadataForReason
import os
from utils import RequestedChunkingType, getAllFilesInDirMatchingFormat
from CollectionNames import REASONING_DOCS

COLLECTION_NAME =  REASONING_DOCS
DB_DIR_PATH = "fakedb"#os.path.join("baseline","db")
DOC_DIR_PATH = os.path.join("./all3gppdocsfromrel17and18","CRs","testset")

def parse_nonCR_docs(file_list,db:DBClient):
    db.updateDBFromFileList(file_list,metadata_func=addExtraDocumentWideMetadataForReason,doc_dir=DOC_DIR_PATH,requested_chunking_type=RequestedChunkingType.SECTION)

def parse_CR_docs(file_list,db:DBClient):
    db.updateDBFromFileList(file_list,metadata_func=addExtraDocumentWideMetadataForReason,doc_dir=DOC_DIR_PATH,requested_chunking_type=RequestedChunkingType.CR)

if __name__ == "__main__":
    db = DBClient(embedding_model_name="text-embedding-3-large",collection_name=COLLECTION_NAME,db_dir_path=DB_DIR_PATH)

    #convertAllDocToDocx(DOC_DIR_PATH)
    file_list = getAllFilesInDirMatchingFormat(DOC_DIR_PATH)
    print(file_list)
    #parse_nonCR_docs(file_list,db)
    parse_CR_docs(file_list,db)
    print(db.queryDB("What is 5G?",k=3))

    