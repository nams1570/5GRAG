import sys
sys.path.append("../")

from abc import ABC,abstractmethod
from openai import OpenAI
from settings import config
from controller import Controller
from baseline.simple_rag_controller import SimpleRAGController, EVOLUTION_BENCHMARK_COLL_NAME, CROSS_CONTEXT_BENCHMARK_COLL_NAME
from baseline.chat3GPP_analogue import Chat3GPPAnalogue

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
        self.response_template = '''Answer the following question in about 200 words.
- Provide a clear, easy-to-understand explanation.
- Use the context below only if it is relevant; otherwise rely on general knowledge.
- If you rely on the context or a specific spec, cite references inline.

<context>
{context}
</context>

Question: {input}

Answer:'''
    
    def get_response(self, question):
        response_prompt = self.response_template.format(input=question,context="NO CONTEXT")

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
        response,orig_docs,additional_docs = self.c.runController(prompt=question,history=[],selected_docs=[])
        return response,orig_docs + additional_docs
    
    def get_only_retrieval_results(self,question:str):
        if not self.isDBInitialized:
            self.c.updateContextDB()
            self.isDBInitialized = True
        _,docs = self.c.getOnlyRetrievalResults(prompt=question,history=[])
        return "",docs

    
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

class Chat3GPPAnalogueModel(BaseSystemModel):
    def __init__(self,db_dir_path,isEvol=True):
        if isEvol:
            collection_name = EVOLUTION_BENCHMARK_COLL_NAME
        else:
            collection_name = CROSS_CONTEXT_BENCHMARK_COLL_NAME
        api_key = config["API_KEY"]
        model_name = config["MODEL_NAME"]
        self.c = Chat3GPPAnalogue(db_dir_path=db_dir_path,collection_name=collection_name,api_key=api_key,model_name=model_name)
    
    def get_response(self, question:str):
       response,_ = self.c.runController(question=question,k1=100,k2=5)
       return response

    def get_response_with_docs(self, question:str):
       response,docs = self.c.runController(question=question,k1=100,k2=5)
       return response,docs
    
    def get_only_retrieval_results(self,question:str):
       _,docs = self.c.getOnlyRetrievalResults(question=question,k1=100,k2=5)
       return "",docs

if __name__ == "__main__":
    gpt = GPTSystemModel()
    c = ControllerSystemModel(isDBInitialized=True,doc_dir_path="../data",db_dir_path="../pickles")
    baseline = BaselineSystemModel("../baseline/db",True)
    chat3gpp = Chat3GPPAnalogueModel("../baseline/db",True)

    question = "What is spurious response?"
    print(chat3gpp.get_response(question))