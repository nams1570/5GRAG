import sys
sys.path.append("..")
import json
from settings import config
from SystemModels import ControllerSystemModel, BaselineSystemModel, BaseSystemModel

ORIGINAL_CHUNK_KEY_NAMES = ["context_for_answer"]

def get_retrieved_docs_from_model(model:BaseSystemModel,question:str):
    _,docs =model.get_response_with_docs(question)
    return docs

def get_original_chunk_from_question_obj(question_obj:dict):
    for key in ORIGINAL_CHUNK_KEY_NAMES:
        if question_obj.get(key,None):
            return question_obj[key]
    return ""

def compare_docs_to_chunk(docs,chunk):
    for doc in docs:
        if chunk == doc.page_content or doc.page_content in chunk or chunk in doc.page_content:
            return 1
    return 0

if __name__ == "__main__":
    input_path = "./relevant_questions_weaker_filter2.json"

    with open(input_path,"r") as f:
        input_data = json.load(f)

    #setup models
    deepspecs_model = ControllerSystemModel(isDBInitialized=True,doc_dir_path="../data",db_dir_path="../pickles")
    baseline_model = BaselineSystemModel("../baseline/db",isEvol=True)

    deepspecs_hits,baseline_hits = 0,0 #number of times model retrieves correct chunk
    # for each question_obj, run retrieval and get original chunk
    for i,question_obj in enumerate(input_data):
        org_chunk = get_original_chunk_from_question_obj(question_obj)
        question = question_obj["question"]
        if org_chunk == "":
            raise Exception(f'could not extract original chunk for question {i}')
        #for each model, run retrieval
        deepspecs_docs = get_retrieved_docs_from_model(deepspecs_model,question)
        baseline_docs = get_retrieved_docs_from_model(baseline_model,question)
        #compare results
        deepspecs_hits += compare_docs_to_chunk(deepspecs_docs,org_chunk)
        baseline_hits += compare_docs_to_chunk(baseline_docs,org_chunk)
    
    print("\n\n\n")
    print(f"deepspecs retrieves the original chunk used to get the question {deepspecs_hits} / {len(input_data)} times \n ")
    print(f"baseline retrieves the original chunk used to get the question {baseline_hits} / {len(input_data)} times \n ")


