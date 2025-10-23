from DBClient import DBClient
import os
from utils import convertAllDocToDocx,getAllFilesInDirMatchingFormat, getTokenCount, RequestedChunkingType
from MetadataAwareChunker import addExtraDocumentWideMetadataForReason
from CollectionNames import SPECS_AND_DISCUSSIONS

COLLECTION_NAME = SPECS_AND_DISCUSSIONS
DB_DIR_PATH = os.path.join("baseline","db")
DOC_DIR_PATH = os.path.join("./all3gppdocsfromrel17and18","Release 17")
  
def parse_spec_list(file_list,db: DBClient):
    db.updateDBFromFileList(file_list,metadata_func=addExtraDocumentWideMetadataForReason,doc_dir=DOC_DIR_PATH,requested_chunking_type=RequestedChunkingType.SECTION)
    
if __name__ == "__main__":
    db = DBClient(embedding_model_name="text-embedding-3-large",collection_name=COLLECTION_NAME,db_dir_path=DB_DIR_PATH)

    file_list = getAllFilesInDirMatchingFormat(DOC_DIR_PATH)
    parse_spec_list(file_list,db)
    print(db.queryDB("What is 5G?",k=3))

