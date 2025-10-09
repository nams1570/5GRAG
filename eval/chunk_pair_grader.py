"""For pairs of chunks (chunkA, chunkB) where chunkA references chunkB, 
we see if chunkB is relevant to chunkA"""
import sys
sys.path.append("..")
import json
from settings import config
import time
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def filter_bad_mappings(chunk_to_chunk_dict_list:list[dict]):
    for chunk_to_chunk_dict in chunk_to_chunk_dict_list:
        if chunk_to_chunk_dict['org_doc']['metadata']['section'] == chunk_to_chunk_dict['ref_doc']['metadata']['section'] and\
        chunk_to_chunk_dict['org_doc']['metadata']['docID'] == chunk_to_chunk_dict['ref_doc']['metadata']['docID']:
            continue
        else:
            yield chunk_to_chunk_dict

def parse_chunk_into_str(chunk:dict)->str:
    """chunk is of the form {'metadata':...,'page_content':...}"""
    return f"metadata: {chunk['metadata']}, content: {chunk['page_content']}"

response_template = '''Compare the original chunk to the reference chunk.
original chunk: {org_chunk}
reference chunk: {ref_chunk}
Instructions:
1. See if the reference chunk provides information that would be useful to understand what the original chunk is talking about.
2. Return 'helpful' if the reference chunk's content will provide necessary detail to fully understand what the original chunk's content is talking about.
3. Return 'unhelpful' if the reference chunk's content is not necessary to fully understand what the original chunk is talking about.
4. Respond only with 'helpful' or 'unhelpful'  '''
        
def compare_chunks_for_ref(client,org_chunk,ref_chunk):
    """Compare original chunk, ref_chunk. see if ref chunk is useful for resolving org_chunk."""
    org_chunk_str = parse_chunk_into_str(org_chunk)
    ref_chunk_str = parse_chunk_into_str(ref_chunk)
    response_prompt = response_template.format(org_chunk=org_chunk_str,ref_chunk=ref_chunk_str)
    response = client.chat.completions.create(
        model=config["MODEL_NAME"],
        messages=[
            {
                'role': 'user',
                'content': response_prompt
            }
        ],
        seed=0,
    )
    return response.choices[0].message.content

def process_item(client,item, max_retries=3, delay=3):
    org_doc,ref_doc = item['org_doc'],item['ref_doc']
    print(f"org doc is {org_doc}\n\n\n")
    for attempt in range(1, max_retries + 1):
        try:
            resp = compare_chunks_for_ref(client,org_chunk=org_doc,ref_chunk=ref_doc)
            return {
                'org_doc':parse_chunk_into_str(org_doc),
                'ref_doc': parse_chunk_into_str(ref_doc),
                'judgment':resp
            }
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                print(f"Max retries reached, skipping this item.")
                return {
                    'org_doc':parse_chunk_into_str(org_doc),
                    'ref_doc': parse_chunk_into_str(ref_doc),
                    'judgment':None
                }
            
def get_injected_prompt(org_chunk,ref_chunk):
    org_chunk_str = parse_chunk_into_str(org_chunk)
    ref_chunk_str = parse_chunk_into_str(ref_chunk)
    response_prompt = response_template.format(org_chunk=org_chunk_str,ref_chunk=ref_chunk_str)
    return response_prompt

def process_item_into_prompt(client,item):
    org_doc,ref_doc = item['org_doc'],item['ref_doc']
    resp = get_injected_prompt(org_chunk=org_doc,ref_chunk=ref_doc)
    return {
        'prompt':resp,
        'org_doc':org_doc,
        'ref_doc': ref_doc,
    }


if __name__ == "__main__":
    file_path = "./all_chunk_ref_pairs.json"
    PROMPT_ONLY = True
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    i = 0
    formatted_data = []
    other_data = []
    """for k in data:
        #for org_to_ref_map in filter_bad_mappings(data[k]):
            formatted_data.append(org_to_ref_map)
        for mmap in data[k]:
            other_data.append(mmap)
        i+=1"""
    for item in data:
        formatted_data.append(item)
    print(len(formatted_data))
    print(len(other_data))
    

    client = OpenAI(
        api_key=config["API_KEY"],
        timeout=60,
    )

    results = [None] * len(formatted_data)

    with ThreadPoolExecutor(max_workers=30) as executor:
        # Submit each item and store a mapping from Future -> index
        future_to_index = {}
        for i, item in enumerate(formatted_data):
            if not PROMPT_ONLY:
                future = executor.submit(process_item, client, item)
            else:
                future = executor.submit(process_item_into_prompt,client,item)
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
                with open("grade_chunks.json", 'w') as f:
                    json.dump(results, f, indent=4)
        
    # Write final results to disk
    with open("grade_chunks.json", 'w') as f:
        json.dump(results, f, indent=4)
