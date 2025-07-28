import sys
sys.path.append("../")

from settings import config
import os
import json
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import time


response_template = '''# Task

You are an expert in QA evaluation for 5G domain tasks. Your task is to determine whether a model-generated answer is accurate or not, given the following:

- **Context:** A passage that provides all the necessary information to answer the question
- **Question:** The question being asked.
- **Reference Answer:** A high-quality human-written answer for reference.

# Context

{context}

# Question

{question}

# Reference Answer

{reference_answer}

# Model Answer

{model_answer}

# Instructions

1. Read the context and question carefully.
2. Compare the model answer to the reference answer and context, focusing on:
    - **Accuracy:** Does the answer match the meaning of the reference answer, based on the given context?
    - **Completeness:** Is all relevant information from the context present in the model answer
    - **Faithfulness to Context:** Is the model answer based only on the given context, without involving external knowledge?
3. Determine whether the model answer is accurate overall, considering all 3 criteria.
4. Respond in the following format:

{{"Reasoning": <Your brief reasoning. Mention accuracy, completeness, and faithfulness to context.>,
"Judgment": <"Accurate" or "Inaccurate">}}'''

def get_response(client,question,ground_truth,model_answer,tdocChunk, seed):
    response_prompt = response_template.format(question=question,reference_answer=ground_truth,model_answer=model_answer,context=tdocChunk)
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
    question = item['question']
    ground_truth = item['ground_truth']
    model_answer = item['predicted_answer']
    tdocChunk = item['context_for_answer']
    for attempt in range(1, max_retries + 1):
        try:
            gpt_response = get_response(client, question,ground_truth,model_answer=model_answer,tdocChunk=tdocChunk, seed=seed)
            response_obj = json.loads(gpt_response)
            return {
                'question': question,
                'ground_truth':ground_truth,
                'predicted_answer':model_answer,
                'judgment':response_obj['Judgment'],
                'reasoning':response_obj['Reasoning']
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
                    'predicted_answer':model_answer,
                    'completion': None,
                }

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', '-s', type=int, default=0)
    argparser.add_argument('--input_path', '-i', type=str)
    argparser.add_argument('--output_path', '-o', type=str)
    args = argparser.parse_args()

    # Load your test input
    with open(args.input_path) as f:
        test_input = json.load(f)

    # Create the client
    client = OpenAI(
        api_key=config["API_KEY"],
        timeout=60,
    )

    # Prepare list to hold final results
    results = [None] * len(test_input)

    # Use ThreadPoolExecutor to process up to 60 items at once
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit each item and store a mapping from Future -> index
        future_to_index = {}
        for i, item in enumerate(test_input):
            future = executor.submit(process_item, client, item, args.seed)
            future_to_index[future] = i

        # As each future finishes, insert its result into `results`
        n_finished = 0
        for future in tqdm(as_completed(future_to_index),
                           total=len(future_to_index),
                           desc="Processing"):
            i = future_to_index[future]
            result = future.result()
            results[i] = result
            n_finished += 1

            # Write partial results to disk after each item completes
            if n_finished == 1 or n_finished % 100 == 0:
                with open(args.output_path, 'w') as f:
                    json.dump(results, f, indent=4)
        
    # Write final results to disk
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)