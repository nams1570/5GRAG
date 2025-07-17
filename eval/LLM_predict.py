import sys
sys.path.append("../")
import time
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from SystemModels import BaseSystemModel,GPTSystemModel, ControllerSystemModel

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
                'predicted_answer':resp
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

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path','-i',type=str)
    argparser.add_argument('--output_path','-o',type=str)
    argparser.add_argument('--use-system',action='store_true',help="pass this argument to use 5grag")
    args = argparser.parse_args()

    if args.use_system:
        system = ControllerSystemModel(isDBInitialized=True,doc_dir_path="../data",db_dir_path="../pickles")
        print("USIN SYSTEM")
    else:
        system = GPTSystemModel()

    with open(args.input_path,"r") as f:
        input_data = json.load(f)

    results = [None]*len(input_data)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_index = {}
        for i,item in enumerate(input_data):
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
        