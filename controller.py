from settings import config
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser #converts output into string
from langchain_core.prompts import ChatPromptTemplate 
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader
import pytesseract
import os


API_KEY = config["API_KEY"]
M_NAME = config["MODEL_NAME"]
DOC_DIR = config["DOC_DIR"]

class Controller:
    def __init__(self):
        self.output_parser = StrOutputParser()
        self.llm = ChatOpenAI(api_key = API_KEY, model=M_NAME)
        self.prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}""")
        self.docs = None

    def updateDocs(self):
        #We need a separate loader for each document. 
        loaders = [UnstructuredFileLoader(os.path.join(DOC_DIR,fn),mode='elements') for fn in os.listdir(DOC_DIR)]
        print(loaders)
        self.docs = []

        for loader in loaders:
            print(f"Document is {loader.file_path}")
            print(os.path.exists(loader.file_path))
            raw_doc = loader.load()
            print(raw_doc[:5])
            text_splitter = RecursiveCharacterTextSplitter() #creates a text splitter, which breaks apart the document into text
            doc = text_splitter.split_documents(raw_doc) #applies the text splitter to the documents
            self.docs += [doc]

    def createVectorStore(self):
        """ create the vector database which will store the vector embeddings of the\
              documents that will be retrieved."""
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small',api_key=API_KEY) #Since we're using openAI's llm, we have to use its embedding model
        self.updateDocs()
        vector = Chroma.from_documents(self.docs, embeddings) 
        self.retriever = vector.as_retriever()
        
    def runController(self):
        self.createVectorStore()
        while True:
            print(f"Ctrl + C to exit...")
            res = input("Please enter your query here:")
            #doc_chain is a chain that lets you pass a document to the llm and it uses that to answer
            doc_chain = create_stuff_documents_chain(self.llm, self.prompt)
            # retrieval chain passed the load of deciding what document to use to answer to the retriever.
            retrieval_chain = create_retrieval_chain(self.retriever, doc_chain)

            resp = retrieval_chain.invoke({"input":res})
            print(f"response is {resp['answer']}")
    
if __name__ == "__main__":
    c = Controller()
    c.runController()


