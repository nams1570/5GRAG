from settings import config
from DBClient import DBClient
from AutoFetcher import AutoFetcher
from utils import unzipFile,convertAllDocToDocx, getTokenCount,RetrieverResult
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser #converts output into string
from langchain_core.prompts import ChatPromptTemplate 
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from MultiStageRetriever import MultiStageRetriever
from sys import getsizeof

API_KEY = config["API_KEY"]
M_NAME = config["MODEL_NAME"]
DOC_DIR = config["DOC_DIR"]
DB_DIR = config["CHROMA_DIR"]
SPEC_COLL_NAME = config["SPEC_COLL_NAME"]
TDOC_COLL_NAME = config["TDOC_COLL_NAME"]
DIFF_COLL_NAME = config["DIFF_COLL_NAME"]

class Controller:
    def __init__(self,doc_dir_path=DOC_DIR,db_dir_path=DB_DIR):
        self.output_parser = StrOutputParser()
        self.llm = ChatOpenAI(api_key = API_KEY, model=M_NAME)

        self.prompt_template = ChatPromptTemplate.from_template("""Answer the following question in 200 words with reference to the provided context:
<context>
{context}
</context>
Question: {input}""")
        self.document_prompt = ChatPromptTemplate.from_template("""
        Page content: {page_content} \n
        From clause: {section}
        """)
        self.doc_chain = create_stuff_documents_chain(self.llm, self.prompt_template, document_prompt=self.document_prompt)

        embeddings = OpenAIEmbeddings(model='text-embedding-3-large',api_key=API_KEY) #Since we're using openAI's llm, we have to use its embedding model
        self.contextDB = DBClient(embedding_model=embeddings,db_dir_path=db_dir_path,doc_dir_path=doc_dir_path)
        self.reasonDB = DBClient(embedding_model=embeddings,collection_name=TDOC_COLL_NAME,db_dir_path=db_dir_path,doc_dir_path="reasoning")
        self.diffDB = DBClient(embedding_model=embeddings,collection_name=DIFF_COLL_NAME,db_dir_path=db_dir_path,doc_dir_path="testchange")

        #endpoints = ["https://www.3gpp.org/ftp/Specs/latest/Rel-16/38_series","https://www.3gpp.org/ftp/Specs/latest/Rel-17/38_series"]
        #endpoints += ["https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series"]
        endpoints = ["https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series/38211-i70.zip","https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series/38212-i70.zip","https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series/38213-i70.zip","https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series/38214-i70.zip"]
        #endpoints += ["https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series/38331-i51.zip","https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series/38181-i50.zip","https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series/38133-i90.zip","https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series/38321-i50.zip","https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series/38174-i70.zip","https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series/38300-i50.zip","https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series/38104-i90.zip","https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series/38175-i10.zip","https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series/38113-i40.zip","https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series/38355-i50.zip","https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series/38108-i60.zip","https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series/38106-i80.zip","https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series/38114-i40.zip"]
        #ORAN Docs
        """print("38 series")
        endpoints += ["https://www.3gpp.org/ftp/Specs/archive/38_series/38.300","https://www.3gpp.org/ftp/Specs/archive/38_series/38.401","https://www.3gpp.org/ftp/Specs/archive/38_series/38.321","https://www.3gpp.org/ftp/Specs/archive/38_series/38.322","https://www.3gpp.org/ftp/Specs/archive/38_series/38.323","https://www.3gpp.org/ftp/Specs/archive/38_series/38.331",]
        print("23 series")
        endpoints += ["https://www.3gpp.org/ftp/Specs/archive/23_series/23.501","https://www.3gpp.org/ftp/Specs/archive/23_series/23.502"]
        print("f1 interface")
        endpoints += ["https://www.3gpp.org/ftp/Specs/archive/38_series/38.470","https://www.3gpp.org/ftp/Specs/archive/38_series/38.473"]
        print("e1 interface")
        endpoints += ["https://www.3gpp.org/ftp/Specs/archive/38_series/38.460","https://www.3gpp.org/ftp/Specs/archive/38_series/38.463"]
        print("xn interface")
        endpoints += ["https://www.3gpp.org/ftp/Specs/archive/38_series/38.420","https://www.3gpp.org/ftp/Specs/archive/38_series/38.423"]
        print("ng interface")
        endpoints += ["https://www.3gpp.org/ftp/Specs/archive/38_series/38.410","https://www.3gpp.org/ftp/Specs/archive/38_series/38.413"]"""
        self.params = params = {"sortby":"date"}
        self.af = AutoFetcher(endpoints,unzipFile,doc_dir_path)

        otherEndpoints = ["https://www.3gpp.org/ftp/TSG_RAN/WG2_RL2/TSGR2_01/Docs/zips"]
        self.afReason = AutoFetcher(otherEndpoints,unzipFile,doc_dir_path)

        self.retriever = MultiStageRetriever(specDB=self.contextDB,discussionDB=self.reasonDB,diffDB=self.diffDB)

        self.isDatabaseTriggered = True

    def updateContextDB(self):
        """Scan dir for new docs and add them"""
        #Fetch new docs
        print(f"resyncing on controller end")
        file_list = self.af.run(self.params,areEndpointsGettable=config["ARE_ENDPOINTS_GETTABLE"])
        file_list = [file[:-4] + ".docx" for file in file_list]
        #split & break down new docs

        #update chroma
        self.contextDB.updateDB(file_list)
        print(f"done resyncing")
    
    def updateReasonDB(self):
        """Fetches latest tdocs and reads into the reason collection"""
        print(f"Hit the update reason!")
        file_list = self.afReason.run()
        convertAllDocToDocx(DOC_DIR)
        file_list = [file[:-4] + ".docx" for file in file_list]
        
        self.reasonDB.updateDB(file_list)
        print("updated collection!")

    def toggleDatabase(self):
        """Switches from RAG mode to non-RAG mode"""
        self.isDatabaseTriggered = not self.isDatabaseTriggered
        if self.isDatabaseTriggered:
            self.prompt_template = ChatPromptTemplate.from_template("""Answer the following question with reference to the provided context:
<context>
{context}
</context>
Question: {input}""")
            self.doc_chain = create_stuff_documents_chain(self.llm,self.prompt_template,document_prompt=self.document_prompt)
        else:
            self.prompt_template = ChatPromptTemplate.from_template("""Answer the following question as best you can Question: {input}""")
        return self.isDatabaseTriggered

    def convert_history(self, history):
        """This turns the 'history' of the frontend into a particular format, separating the human and ai messages."""
        message_objects = []
        for turn in history:
            message_objects.append(HumanMessage(content=turn[0]))
            message_objects.append(AIMessage(content=turn[1]))
        return message_objects
    

    def getResponseWithRetrieval(self,prompt,history):
        retriever_result:RetrieverResult = self.retriever.invoke(query=prompt)

        retrieved_docs = retriever_result.firstOrderSpecDocs + retriever_result.secondOrderSpecDocs + retriever_result.retrievedDiscussionDocs

        resp_answer = self.doc_chain.invoke({"context":retrieved_docs,"input":prompt,"history":history})
        resp = {"input":prompt,"history":history,"context":retrieved_docs,"answer":resp_answer}
        print(f"size of answer is {getsizeof(resp_answer)} and token count is {getTokenCount(resp_answer,M_NAME)}")
        return resp,retriever_result.firstOrderSpecDocs,retriever_result.secondOrderSpecDocs

    def runController(self, prompt:str, history:list[list[str]], selected_docs:list[str]):
        print('Selected Docs: ', selected_docs)
        self.retriever.constructRetriever(db=self.contextDB,selected_docs=selected_docs)

        if prompt:
            print(f"Ctrl + C to exit...")
            #doc_chain is a chain that lets you pass a document to the llm and it uses that to answer
            # retrieval chain passed the load of deciding what document to use to answer to the retriever.
            history = self.convert_history(history)
            if self.isDatabaseTriggered:
                resp,orig_docs,additional_docs = self.getResponseWithRetrieval(prompt,history)
                print(f"resp is {resp}")
                response = resp['answer']
            else:
                chain = self.prompt_template | self.llm
                resp = chain.invoke({"input":prompt,"history": history})
                response = resp.content
                orig_docs,additional_docs = [],[]
            return response,orig_docs,additional_docs
    
if __name__ == "__main__":
    c = Controller()
    c.runController()


