from settings import config
from utils import getTokenCount,RetrieverResult
from MultiStageRetriever import MultiStageRetriever
from RAGQAEngine import RAGQAEngine
from CollectionNames import SPECS_AND_DISCUSSIONS as SPEC_COLL_NAME, REASONING_DOCS as TDOC_COLL_NAME, DIFFS as DIFF_COLL_NAME, CROSS_CONTEXT_BENCHMARK_COLL_NAME
from sys import getsizeof

API_KEY = config["API_KEY"]
M_NAME = config["MODEL_NAME"]
DOC_DIR = config["DOC_DIR"]
DB_DIR = config["CHROMA_DIR"]

class Controller:
    def __init__(self,db_dir_path=DB_DIR):
        self.qa_engine = RAGQAEngine(prompt_template_file_path="prompt.txt",model_name=M_NAME,api_key=API_KEY)

        self.retriever = MultiStageRetriever(pathToDB=db_dir_path,specCollectionName=SPEC_COLL_NAME,reasonCollectionName=TDOC_COLL_NAME,diffCollectionName=DIFF_COLL_NAME)

        self.isDatabaseTriggered = True

    def toggleDatabase(self):
        """Switches from RAG mode to non-RAG mode"""
        self.isDatabaseTriggered = not self.isDatabaseTriggered
        return self.isDatabaseTriggered

    def getResponseWithRetrieval(self,prompt):
        retriever_result:RetrieverResult = self.retriever.invoke(query=prompt)

        retrieved_docs = retriever_result.firstOrderSpecDocs + retriever_result.secondOrderSpecDocs + retriever_result.retrievedDiscussionDocs

        resp_answer = self.qa_engine.get_answer_from_context(prompt, retrieved_docs)
        resp = {"input":prompt,"context":retrieved_docs,"answer":resp_answer}
        print(f"size of answer is {getsizeof(resp_answer)} and token count is {getTokenCount(resp_answer,M_NAME)}")
        return resp,retriever_result.firstOrderSpecDocs,retriever_result.secondOrderSpecDocs,retriever_result.retrievedDiscussionDocs
    
    def getOnlyRetrievalResults(self,prompt:str)->tuple[str,list]:
        retriever_result:RetrieverResult = self.retriever.invoke(query=prompt)

        retrieved_docs = retriever_result.firstOrderSpecDocs + retriever_result.secondOrderSpecDocs + retriever_result.retrievedDiscussionDocs

        return "",retrieved_docs

    def runController(self, prompt:str, selected_docs:list[str]):
        print('Selected Docs: ', selected_docs)

        if prompt:
            print(f"Ctrl + C to exit...")
            if self.isDatabaseTriggered:
                resp,orig_docs,additional_docs,discussion_docs = self.getResponseWithRetrieval(prompt)
                print(f"resp is {resp}")
                response = resp['answer']
                additional_docs += discussion_docs
            else:
                response = self.qa_engine.get_raw_answer(prompt)
                orig_docs,additional_docs = [],[]
            return response,orig_docs,additional_docs
    
if __name__ == "__main__":
    c = Controller()
    c.runController()


