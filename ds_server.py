from settings import config
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import threading
from controller import Controller
from DBClient import DBClient
from CollectionNames import DIFFS as DIFF_COLL_NAME
from utils import Document
import datetime

ds_controller = Controller()
db_client = DBClient(collection_name=DIFF_COLL_NAME,db_dir_path=config["CHROMA_DIR"])

class QARequest(BaseModel):
    question: str

class ClauseHistoryRequest(BaseModel):
    clause_id: str

def _serialize_document_list(docs:list[Document]) -> list[dict]:
    return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]

################
# Rate Limiter #
################
# Rate Limiter #
################
_rate_lock = threading.Lock()
_session_counts: dict[str,int] = {}
_current_day_count = 0
_last_daily_reset = datetime.date.today()

MAX_PER_SESSION = config.get("MAX_REQUESTS_PER_SESSION", 100)
MAX_DAILY_REQUESTS = config.get("MAX_DAILY_REQUESTS", 300)

def is_under_rate_limit(request: Request)->bool:
    """Returns True if under rate limit, False otherwise. Must increment counts if under limit in a thread safe manner. Must reset the global variables at the start of a new day."""
    global _last_daily_reset, _current_day_count
    try:
        client_ip = request.client.host or "unknown"
    except Exception:
        client_ip = "unknown"
    
    today = datetime.date.today()
    with _rate_lock:
        if today > _last_daily_reset:
            _last_daily_reset = today
            _current_day_count = 0
            _session_counts.clear()

        cur_session_count = _session_counts.get(client_ip, 0)
        if cur_session_count + 1 >= MAX_PER_SESSION:
            return False

        if _current_day_count + 1 >= MAX_DAILY_REQUESTS:
            return False
        
        _session_counts[client_ip] = cur_session_count + 1
        _current_day_count += 1

    return True

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/diffs")
def read_diffs(clause_history_request: ClauseHistoryRequest, request: Request):
    filter = {'section':{"$eq":clause_history_request.clause_id}}
    try:
        docs = db_client.queryDB(query_text="*",k=10,filter=filter)
        serialized_docs = _serialize_document_list(docs)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return JSONResponse(status_code=200, content={"clause_id": clause_history_request.clause_id, "documents": serialized_docs})

@app.post("/qa")
def read_qa(qa_item: QARequest, request: Request):
    if not is_under_rate_limit(request):
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded. Try again in a day"})
    question = qa_item.question
    try:
        resp, orig_docs, additional_docs = ds_controller.runController(question)
    except Exception as e:
        return JSONResponse(status_code=500, content={"question": question, "error": str(e)})
    return JSONResponse(status_code=200, content={"question": question, "answer": resp, "original_documents": _serialize_document_list(orig_docs), "additional_documents": _serialize_document_list(additional_docs)})
