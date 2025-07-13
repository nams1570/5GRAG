import sys
sys.path.append("..")
from settings import config
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json

response_template = '''You will be given a question, the gold truth answer to that question, and a chunk of text.
Respond with 'No' if you can answer the question correctly using just the chunk given.
Respond with 'Yes' if you need more context than just the chunk given to answer the question correctly.
question: {question} \n
gold truth answer: {answer}\n
chunk: {chunk} \n
Instructions:
1. Respond only with Yes or No. One word only.'''
def get_response(client,chunk, question, answer, seed):

    response_prompt = response_template.format(question=question,answer=answer,chunk=chunk)
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
    chunk,question,answer = item["primary_chunk_text"],item["question"],item["ground_truth"]
    for attempt in range(1, max_retries + 1):
        try:
            gpt_response = get_response(client,chunk,question,answer, seed)
            response = gpt_response
            return {
                'question':question,
                'ground_truth':answer,
                'is_good_question':response,
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
                    'is_good_question': None,
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

    final_results = []

    for result in results:
        if result["is_good_question"] == "Yes":
            final_results.append(result)
    
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=4)
