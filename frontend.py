import gradio as gr

from controller import Controller

ds_controller = Controller()

def respond(prompt,history):
    """@prompt: question to the model
    @history: the history of the conversation, stored in "turns" of HumanMessage,AIMessage"""
    resp,orig_docs,additional_docs = ds_controller.runController(prompt)
    history.append((prompt,resp))

    latest = [(prompt, resp)]
    return latest,orig_docs,additional_docs


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
    toggleDB = gr.Button("RAG ENABLED CURRENTLY")

    toggleDB.click(fn=adjustToggle,outputs = toggleDB)
    
    chatbot=gr.Chatbot(height=500)
    textbox=gr.Textbox(placeholder="Ask me any question", container=False, scale=7)

    retrieved_orig_docs = gr.Textbox(label="Chunks that were originally retrieved")
    retrieved_additional_docs = gr.Textbox(label="Chunks that were secondarily retrieved, via deep context")

    textbox.submit(respond,inputs=[textbox,chatbot],outputs=[chatbot,retrieved_orig_docs,retrieved_additional_docs])

demo.launch(share=True)