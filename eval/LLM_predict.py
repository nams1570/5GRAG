from ast import arg
import sys
sys.path.append("../")
import time
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from SystemModels import BaseSystemModel,GPTSystemModel, ControllerSystemModel, BaselineSystemModel, Chat3GPPAnalogueModel

def get_other_keys(item:dict):
    """get all the other keys of the dict that aren't explicitly delineated"""
    other_keys = set(item.keys()) - {'question','ground_truth'}
    return {key:item[key] for key in other_keys}

def process_item(system:BaseSystemModel,item:dict,max_retries:int=3,delay:int=3)->dict:
    """This function represents the system model processing a single item"""
    question = item['question']
    ground_truth = item['ground_truth']
    for attempt in range(1,max_retries+1):
        try:
            resp = system.get_response(question)
            return {
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer':resp,
                **get_other_keys(item)
            }
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                print(f"Max retries reached, skipping this item.")
                return {
                    'question': question,
                    'ground_truth':ground_truth,
                    'predicted_answer':None,
                }
            
def process_item_with_docs(system:BaseSystemModel,item:dict,max_retries:int=3,delay:int=3)->dict:
    """This function represents the system model processing a single item"""
    question = item['question']
    ground_truth = item['ground_truth']
    for attempt in range(1,max_retries+1):
        try:
            resp,docs = system.get_response_with_docs(question)
            return {
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer':resp,
                'retrieved_docs':[f"{{metadata: {doc.metadata},\n content: {doc.page_content} }}" for doc in docs],
                **get_other_keys(item)
            }
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                print(f"Max retries reached, skipping this item.")
                return {
                    'question': question,
                    'ground_truth':ground_truth,
                    'predicted_answer':None,
                    'retrieved_docs':None
                }
def process_item_only_retrieval(system:BaseSystemModel,item:dict,max_retries:int=3,delay:int=3)->dict:
    """Process item to only retrieve documents"""
    question = item['question']
    ground_truth = item['ground_truth']
    for attempt in range(1,max_retries+1):
        try:
            _,docs = system.get_only_retrieval_results(question)
            return {
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer':"",
                'retrieved_docs':[f"{{metadata: {doc.metadata},\n content: {doc.page_content} }}" for doc in docs],
                **get_other_keys(item)
            }
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                print(f"Max retries reached, skipping this item.")
                return {
                    'question': question,
                    'ground_truth':ground_truth,
                    'predicted_answer':None,
                    'retrieved_docs':None
                }

def are_passing_multiple_models_in_args(args):
    return (args.use_system and args.use_baseline) or (args.use_system and args.use_3gpp) or (args.use_baseline and args.use_3gpp)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path','-i',type=str)
    argparser.add_argument('--output_path','-o',type=str)
    argparser.add_argument('--use-system',action='store_true',help="pass this argument to use deepspecs")
    argparser.add_argument('--use-baseline',action='store_true',help='pass this argument to use the baseline')
    argparser.add_argument('--use-3gpp',action='store_true',help='pass this argument to use chat3gpp analogue')
    argparser.add_argument('--retrieve-docs',action='store_true',help='pass this argument to also retrieve the docs used in generating the answer')
    argparser.add_argument('--only-ret',action='store_true',help='pass this argument to only retrieve the docs used in generating the answer')
    args = argparser.parse_args()

    if are_passing_multiple_models_in_args(args):
        raise Exception("Error: Can't use multiple models at once")

    if args.use_system:
        system = ControllerSystemModel(isDBInitialized=True,doc_dir_path="../data",db_dir_path="../baseline/db")
        print("USING SYSTEM")
    elif args.use_baseline:
        system = BaselineSystemModel("../baseline/db",isEvol=True)
        print("USING Baseline")
    elif args.use_3gpp:
        system = Chat3GPPAnalogueModel("../baseline/db",isEvol=True)
        print("USING Chat3gpp")
    else:
        if args.retrieve_docs:
            raise Exception("Error: Must pass one of --use-system, --use-baseline, or --use-3gpp if also passing --retrieve-docs")
        if args.only_ret:
            raise Exception("Error: Must pass one of --use-system, --use-baseline, or --use-3gpp if also passing --only-ret")
        system = GPTSystemModel()

    with open(args.input_path,"r",encoding="utf-8",errors="replace") as f:
        input_data = json.load(f)

    results = [None]*len(input_data)
    
    with ThreadPoolExecutor(max_workers=30) as executor:
        future_to_index = {}
        for i,item in enumerate(input_data):
            if args.retrieve_docs:
                print("Retrieving docs for item ")
                future = executor.submit(process_item_with_docs,system,item)
            elif args.only_ret:
                print("Only retrieving docs for item ")
                future = executor.submit(process_item_only_retrieval,system,item)
            else:
                future = executor.submit(process_item,system,item)
            future_to_index[future] = i
        
        n_finished = 0
        for future in tqdm(as_completed(future_to_index),total = len(future_to_index), desc="Processing..."):
            i = future_to_index[future]
            result = future.result()
            results[i] = result
            n_finished +=1

            if n_finished == 1 or n_finished %100 !=0:
                with open(args.output_path,"w") as f:
                    json.dump(results,f,indent=4)
        
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)
        