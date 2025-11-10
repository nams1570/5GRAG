import gradio as gr
import os
from dotenv import load_dotenv
import requests

load_dotenv()

USE_REMOTE_DS = os.getenv("USE_REMOTE_DS", "False").lower() in ["true","1"]
DS_SERVER_URL = os.getenv("DS_SERVER_URL", "http://localhost:8000")


print(f"use remote ds is {USE_REMOTE_DS}, and {DS_SERVER_URL}")
if not USE_REMOTE_DS:
    from controller import Controller
    ds_controller = Controller()

def respond_via_remote(prompt,history):
    try:
        payload = {"question": prompt}
        resp = requests.post(DS_SERVER_URL+"/qa", json=payload)
        resp.raise_for_status()
        data = resp.json()
        answer,orig_docs,additional_docs = data.get("answer"),data.get("original_documents",[]),data.get("additional_documents",[])
        history.append((prompt,answer))
        latest = [(prompt, answer)]
        return latest, orig_docs, additional_docs
    except requests.RequestException as e:
        raise ValueError(f"Error communicating with remote DS: {e}")

def respond_locally(prompt,history):
    resp,orig_docs,additional_docs = ds_controller.runController(prompt)
    history.append((prompt,resp))

    latest = [(prompt, resp)]
    return latest,orig_docs,additional_docs

def respond(prompt,history):
    """@prompt: question to the model
    @history: the history of the conversation, stored in "turns" of HumanMessage,AIMessage"""
    if USE_REMOTE_DS:
        return respond_via_remote(prompt, history)
    else:
        return respond_locally(prompt, history)


def adjustToggle():
    """This function controls whether RAG/ Vector DB is used or not.
    It keeps the controller and the frontend in sync by coupling them."""
    IS_DB = ds_controller.toggleDatabase()
    if IS_DB:
        text = "RAG ENABLED CURRENTLY"
    else:
        text = "RAG DISABLED CURRENTLY"
    return text

# idea: design function for the submit. This function will have as inputs, prompt and history
# get history from chatbot. As output, have "", history so both q and answer will go into chatbot

with gr.Blocks() as demo:
    if not USE_REMOTE_DS:
        toggleDB = gr.Button("RAG ENABLED CURRENTLY")

        toggleDB.click(fn=adjustToggle,outputs = toggleDB)
    
    chatbot=gr.Chatbot(height=500)
    textbox=gr.Textbox(placeholder="Ask me any question", container=False, scale=7)

    retrieved_orig_docs = gr.Textbox(label="Chunks that were originally retrieved")
    retrieved_additional_docs = gr.Textbox(label="Chunks that were secondarily retrieved, via deep context")

    textbox.submit(respond,inputs=[textbox,chatbot],outputs=[chatbot,retrieved_orig_docs,retrieved_additional_docs])

demo.launch(share=True)