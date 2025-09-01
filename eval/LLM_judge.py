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

response_template = '''Evaluate the accuracy of a provided answer to a telecommunications and networking question. 
Question: {question}
Ground Truth Answer: {ground_truth}
Provided Answer: {prediction}
Instructions:
1. Compare Provided Answer to Ground Truth Answer. 
2. A provided answer is 'Accurate' if it covers all the same points as the ground truth answer. 
3. A provided answer is 'Inaccurate' if it misses out on any of the points that the ground truth answer covers.
4. The confidence level field should be populated by only the number.
5. Respond in the following format:

{{"Reasoning": <Your brief reasoning on why the judgment should be accurate or inaccurate>,
"Judgment": <"Accurate" or "Inaccurate">,
"Confidence level":<How confident you feel about the judgment. It should be on a scale of 1-5, where 5 means absolutely confident and 1 means not confident at all.> }}
Is the Provided Answer Correct?'''
def get_response(client,question,ground_truth,prediction, seed):
    response_prompt = response_template.format(question=question,ground_truth=ground_truth,prediction=prediction)
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
    prediction = item['predicted_answer']
    for attempt in range(1, max_retries + 1):
        try:
            gpt_response = get_response(client, question,ground_truth,prediction, seed)
            response_obj = json.loads(gpt_response)
            return {
                'question': question,
                'ground_truth':ground_truth,
                'predicted_answer':prediction,
                'judgment': response_obj['Judgment'],
                'reasoning': response_obj['Reasoning'],
                'confidence_level': response_obj['Confidence level'],
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
                    'ground_truth':ground_truth,
                    'predicted_answer':prediction,
                    'completion': None,
                    **item,
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
