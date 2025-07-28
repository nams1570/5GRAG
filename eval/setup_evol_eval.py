import sys
sys.path.append("..")
import os
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from utils import getAllFilesInDirMatchingFormat
from MetadataAwareChunker import getFullSectionChunks,addExtraDocumentWideMetadataForReason
from settings import config
from ChangeTracker import ChangeTracker, get_empty_document

DIFF_DOC_DIR = "../testchange"


#get diffs from docs in testchange.

def get_diff_list():
    """Returns: a list of strings where each string represents a change made"""

    diff_list = []

    versionToSectionToChunk = {}
    versionToMetadata = {}
    file_list = getAllFilesInDirMatchingFormat(DIFF_DOC_DIR)
    for file in file_list:
        chunks = getFullSectionChunks([os.path.join(DIFF_DOC_DIR,file)],addExtraDocumentWideMetadataForReason)
        version = chunks[0].metadata["version"]
        versionToSectionToChunk[version] = {}
        versionToMetadata[version] = chunks[0].metadata
        for chunk in chunks:
            versionToSectionToChunk[version][chunk.metadata["section"]] = chunk
        
    for fromVersion in versionToSectionToChunk.keys():
        for toVersion in versionToSectionToChunk.keys():
            if ChangeTracker.areAdjacentVersions(fromVersion=fromVersion,toVersion=toVersion):
                for section in versionToSectionToChunk[toVersion].keys():
                    defaultFromChunk = get_empty_document(metadata={**versionToMetadata[fromVersion],'section':section})
                    fromChunk = versionToSectionToChunk[fromVersion].get(section,defaultFromChunk)
                    toChunk = versionToSectionToChunk[toVersion][section]
                    
                    change_obj = ChangeTracker.getChanges(fromChunk,toChunk)
                    for change in change_obj['changes']['add']:
                        diff_list.append({'metadata':change_obj['metadata'],'text':change})
    return diff_list

    

#ask llm to come up with a question from the diff. Question should be about why that change was introduced
question_gen_template = '''Here is a chunk of text that represents the change between two versions of a 3GPP Technical Specification.
Come up with a question that asks about why this change is made or why the new feature in the change is supported. 
The change: {diff_text}
Instructions:
1. The question should not be answerable with only the text of the change.
2. Respond only with the question.'''
def get_response_question_generation(client,diff_text,seed):

    response_prompt = question_gen_template.format(diff_text=diff_text)
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


# for each diff, grab the text of the add change and give it to the llm.
def get_question_from_diff(client,diff:dict,seed, max_retries=3, delay=3):
    """Encapsulates the logic for processing a single item."""
    for attempt in range(1, max_retries + 1):
        try:
            gpt_response = get_response_question_generation(client,diff["text"], seed)
            return {
                'question': gpt_response,
                'fromVersion': diff['metadata']['from_version'],
                'toVersion': diff['metadata']['to_version'],
                'fromTimestamp':diff['metadata']['fromTimestamp'],
                'toTimestamp': diff['metadata']['toTimestamp'],
                'srcContentForQuestion':diff['text'],
                'section':diff['metadata']['section'],
            }
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                print(f"Max retries reached, skipping this item.")
                return {
                    'question': None,
                    'fromVersion': diff['metadata']['from_version'],
                    'toVersion': diff['metadata']['to_version'],
                    'fromTimestamp':diff['metadata']['fromTimestamp'],
                    'toTimestamp': diff['metadata']['toTimestamp'],
                    'srcContentForQuestion':diff['text'],
                    'section':diff['metadata']['section'],
                }
# then, use this created question to query the discussion db
# grab the context from the discussion db
# pass this context int osecond prompt, along with question. Ask it to come up with a ground truth answer

if __name__ == "__main__":
    output_path = "./results.json"

    client = OpenAI(
        api_key=config["API_KEY"],
        timeout=60,
    )

    diff_list = get_diff_list()[:10]
    results = [None] * len(diff_list)
    
    # Use ThreadPoolExecutor to process up to 60 items at once
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit each item and store a mapping from Future -> index
        future_to_index = {}
        for i, diff in enumerate(diff_list):
            future = executor.submit(get_question_from_diff, client, diff, seed=0)
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
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=4)
        
    # Write final results to disk
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)