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
from operator import itemgetter
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
from langchain_community.retrievers import TFIDFRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.load import dumps, loads
import gradio as gr
import os
# Set logging for the queries
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
documents = []
directory = '/Users/harkanwarsingh/Desktop/CS219/5g-specs/'
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000, add_start_index=True)
# Iterate through the directory

#different splitters & loaders
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".pdf"):
            full_path = os.path.join(root, file)
            print(full_path)
            loader = PyPDFLoader(full_path)
            pages =  loader.load_and_split()
            documents.extend(text_splitter.split_documents(pages))
vector = FAISS.from_documents(documents, embeddings)

llm = ChatOpenAI(model="gpt-3.5-turbo-0125",temperature=0.0)

# Different-prompt , different temp
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}\n Analyse the Context and to extract relevant and compile a answer to the question.\nIf the Question ask about changes between 2 version , report the changes compared to previous version.\n Also qoute the paragraph and mention section number for the information used to give answer."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}"),
])

document_chain = create_stuff_documents_chain(llm, prompt)

template = ("Breakdown the Original question given to generate few subqueries need to answer the given questions.\n"
            "For example given a quer to compare a feature on Version 1 vs version 2. Can include subqueries on that feature on Version 1 and Version 2 separately.\n"
            "Original question: {input}")
prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives
    | ChatOpenAI(temperature=0.0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

#retriever = vector.as_retriever(k=10)
#
#retriever_from_llm = MultiQueryRetriever.from_llm(
#    retriever=vector.as_retriever(), llm=llm
#)
retriever2 = TFIDFRetriever.from_documents(documents,k=5)
#retrieval_chain = create_retrieval_chain(retriever2, document_chain)
retrieval_chain = generate_queries | retriever2.map() | get_unique_union

final_rag_chain = (
    {"context": retrieval_chain,
     "history": itemgetter("history"),
     "input": itemgetter("input")}
    | prompt
    | llm
)

def convert_history(history):
  message_objects = []
  for turn in history:
    message_objects.append(HumanMessage(content=turn[0]))
    message_objects.append(AIMessage(content=turn[1]))
  return message_objects


def langchain_server(input,hist):
    history = convert_history(hist)
    print(history)
    
     # Retrieve relevant documents
    print("####QUERIES###########")
    print(generate_queries.invoke(input))
    retrieved_docs = retrieval_chain.invoke(input)
    print("####Retrieved Documents:####")
    for doc in retrieved_docs:
        print(doc)
#    docs_and_scores = vector.similarity_search_with_score(input,k=10)
#    for d in (docs_and_scores):
#        print(d)
#
#    print(TFIDFRetriever.from_documents(retrieved_docs,k=10))
    #response = retrieval_chain.invoke({"input": input,"history": history})
    response = final_rag_chain.invoke({"input": input,"history": history})
    print(response)
    return response.content

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
