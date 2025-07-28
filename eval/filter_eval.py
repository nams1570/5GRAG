import sys
sys.path.append("..")
from settings import config
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
#from LLM_judge import process_item as judge_answer
from evol_rubric_judge import process_item as judge_answer
import time
import json

response_template = '''Answer the question given below in upto 200 words:
question: {question}'''
def get_response(client,question, seed):

    response_prompt = response_template.format(question=question)
    print(response_prompt)
    response = client.chat.completions.create(
        model=config["MODEL_NAME"],
        messages=[
            {
                'role': 'user',
                'content': response_prompt
            }
        ],
        seed=seed,
    )
    return response.choices[0].message.content

def process_item(client, item, seed, max_retries=3, delay=3):
    """Encapsulates the logic for processing a single item."""
    question = item["question"]
    answer = item['ground_truth']
    for attempt in range(1, max_retries + 1):
        try:
            gpt_response = get_response(client,question, seed)
            response = gpt_response
            return {
                'question':question,
                'ground_truth':answer,
                'predicted_answer':response,
                **item,
            }
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                print(f"Max retries reached, skipping this item.")
                return {
                    'question': question,
                    'ground_truth': answer,
                    'predicted_answer': None,
                    **item,
                }

def filter_questions_via_llm(question_objs):
    pass

if __name__ == "__main__":
    client = OpenAI(
        api_key=config["API_KEY"],
        timeout=60,
    )

    input_path = "./results.json"
    output_path = "./filtered_questions.json"

    with open(input_path,"r") as f:
        input_data = json.load(f)

    results = [None]*len(input_data)

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_index = {}
        for i,item in enumerate(input_data):
            future = executor.submit(process_item,client,item,seed=0)
            future_to_index[future] = i
        
        n_finished = 0
        for future in tqdm(as_completed(future_to_index),total = len(future_to_index), desc="Processing..."):
            i = future_to_index[future]
            result = future.result()
            results[i] = result
            n_finished +=1

    final_judgments = [None] * len(results)

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_index = {}
        for i,item in enumerate(results):
            future = executor.submit(judge_answer,client,item,seed=0)
            future_to_index[future] = i
        
        n_finished = 0
        for future in tqdm(as_completed(future_to_index),total = len(future_to_index), desc="Processing..."):
            i = future_to_index[future]
            result = future.result()
            final_judgments[i] = result
            n_finished +=1
    
    final_results = []

    for result in final_judgments:
        if result.get("completion",None) == "No":
            del result["completion"]
            final_results.append(result)
        elif result.get("judgment",None) == "Inaccurate":
            del result["judgment"]
            del result["reasoning"]
            final_results.append(result)
    
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=4)