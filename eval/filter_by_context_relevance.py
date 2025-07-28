import sys
sys.path.append("..")
from settings import config
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json

context_answer_question_prompt = '''Below you are given a question and some context. Does the context answer the question?
<context>
{context}
</context>
Question: {question}
Instructions:
1. Reply only with "Yes" or "No".
2. If the context does contain a clear and specific answer to the question, respond with "Yes"
3. If the context does not contain enough information to answer the question, or if the answer is ambiguous, missing, or requires external knowledge, respond with "No"
'''
def get_response(client,question,context,seed):
    response_prompt = context_answer_question_prompt.format(question=question,context=context)
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
    context = item['context_for_answer']
    for attempt in range(1, max_retries + 1):
        try:
            gpt_response = get_response(client,question,context, seed)
            response = gpt_response
            return {
                'question':question,
                'ground_truth':answer,
                'is_good_question':response,
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


if __name__ == "__main__":
    client = OpenAI(
        api_key=config["API_KEY"],
        timeout=60,
    )

    input_path = "./results.json"
    output_path = "./relevant_questions.json"

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
