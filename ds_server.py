from settings import config
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from controller import Controller
from DBClient import DBClient
from CollectionNames import DIFFS as DIFF_COLL_NAME
from utils import Document

ds_controller = Controller()
db_client = DBClient(collection_name=DIFF_COLL_NAME,db_dir_path=config["CHROMA_DIR"])

class QARequest(BaseModel):
    question: str

class ClauseHistoryRequest(BaseModel):
    clause_id: str

def _serialize_document_list(docs:list[Document]) -> list[dict]:
    return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/diffs")
def read_diffs(clause_history_request: ClauseHistoryRequest):
    filter = {'section':{"$eq":clause_history_request.clause_id}}
    try:
        docs = db_client.queryDB(query_text="*",k=10,filter=filter)
        serialized_docs = _serialize_document_list(docs)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return JSONResponse(status_code=200, content={"clause_id": clause_history_request.clause_id, "documents": serialized_docs})

@app.post("/qa")
def read_qa(qa_item: QARequest):
    question = qa_item.question
    try:
        resp, orig_docs, additional_docs = ds_controller.runController(question)
    except Exception as e:
        return JSONResponse(status_code=500, content={"question": question, "error": str(e)})
    return JSONResponse(status_code=200, content={"question": question, "answer": resp, "original_documents": _serialize_document_list(orig_docs), "additional_documents": _serialize_document_list(additional_docs)})
