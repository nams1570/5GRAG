import sys
sys.path.append("../")

from abc import ABC,abstractmethod
from openai import OpenAI
from settings import config
from controller import Controller
from baseline.simple_rag_controller import SimpleRAGController, EVOLUTION_BENCHMARK_COLL_NAME, CROSS_CONTEXT_BENCHMARK_COLL_NAME

class BaseSystemModel(ABC):
    @abstractmethod
    def get_response(self,input):
        pass

class GPTSystemModel(BaseSystemModel):
    def __init__(self):
        self.client = OpenAI(
            api_key=config["API_KEY"],
            timeout=60,
            )
        self.response_template = '''The following is a question about telecommunications and networking. Just give the answer in about 200 words.
        Question:{question}
        Answer:'''
    
    def get_response(self, question):
        response_prompt = self.response_template.format(question=question)

        response = self.client.chat.completions.create(
        model=config["MODEL_NAME"],
        messages=[
                {
                    'role': 'user',
                    'content': response_prompt
                }
            ],
            seed=0,
        )
        return response.choices[0].message.content
    
class ControllerSystemModel(BaseSystemModel):
    def __init__(self,doc_dir_path:str,db_dir_path:str,isDBInitialized:bool=False):
        self.c = Controller(doc_dir_path=doc_dir_path,db_dir_path=db_dir_path)
        self.isDBInitialized = isDBInitialized

    def get_response(self, question:str):
        if not self.isDBInitialized:
            self.c.updateContextDB()
            self.isDBInitialized = True
        response,_,_ = self.c.runController(prompt=question,history=[],selected_docs=[])
        return response
    
    def get_response_with_docs(self,question:str):
        if not self.isDBInitialized:
            self.c.updateContextDB()
            self.isDBInitialized = True
        response,_,additional_docs = self.c.runController(prompt=question,history=[],selected_docs=[])
        return response,additional_docs

    
class BaselineSystemModel(BaseSystemModel):
    def __init__(self,db_dir_path,isEvol=True):
        if isEvol:
            collection_name = EVOLUTION_BENCHMARK_COLL_NAME
        else:
            collection_name = CROSS_CONTEXT_BENCHMARK_COLL_NAME
        print(f"collection name is {collection_name}")
        api_key = config["API_KEY"]
        model_name = config["MODEL_NAME"]
        self.c = SimpleRAGController(db_dir_path=db_dir_path,collection_name=collection_name,api_key=api_key,model_name=model_name)
    
    def get_response(self,question:str):
        response,_ = self.c.runController(question,k=config["NUM_DOCS_INITIAL_RETRIEVAL"])
        return response

    def get_response_with_docs(self,question:str):
        response,docs = self.c.runController(question,k=config["NUM_DOCS_INITIAL_RETRIEVAL"])
        return response,docs


if __name__ == "__main__":
    gpt = GPTSystemModel()
    c = ControllerSystemModel(isDBInitialized=True,doc_dir_path="../data",db_dir_path="../pickles")
    baseline = BaselineSystemModel("../baseline/db",True)

    question = "What is spurious response?"
    print(baseline.get_response(question))