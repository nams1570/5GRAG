#!/usr/bin/env python
from typing import List

#from fastapi import FastAPI
import bs4
from langchain import hub
import pypdf
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.agents import create_openai_functions_agent
from langchain_core.prompts import MessagesPlaceholder
#from langchain.agents import AgentExecutor
#from langchain.pydantic_v1 import BaseModel, Field
#from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr
import os



embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
documents = []
directory = '/Users/harkanwarsingh/Desktop/CS219/5g-specs/'
text_splitter = RecursiveCharacterTextSplitter()
# Iterate through the directory

#different splitters & loaders
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".pdf"):
            full_path = os.path.join(root, file)
            print(full_path)
            loader = PyPDFLoader(full_path)
            pages = loader.load_and_split()
            documents.extend(text_splitter.split_documents(pages))
vector = FAISS.from_documents(documents, embeddings)

llm = ChatOpenAI(model="gpt-3.5-turbo-0125",temperature=0)

# Different-prompt , different temp
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}"),
])

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def convert_history(history):
  message_objects = []
  for turn in history:
    message_objects.append(HumanMessage(content=turn[0]))
    message_objects.append(AIMessage(content=turn[1]))
  return message_objects


def langchain_server(input,hist):
    history = convert_history(hist)
    print(history)
    response = retrieval_chain.invoke({"input": input,"history": history})
    return response["answer"]

# More improved UI
gr.ChatInterface(
    langchain_server,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask me any question", container=False, scale=7),
    title="5G Expert",
    description="Ask any question",
    theme="soft",
#    examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
#    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch(share=True)
