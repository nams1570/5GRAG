import sys
sys.path.append("..")
import json
from ReferenceExtractor import ReferenceExtractor

RE = ReferenceExtractor()

def print_stats_on_data(data):
    num_helpful, num_unhelpful = 0,0
    uncategorized = []
    for i,item in enumerate(data):
        if item["completion"] == "helpful":
            num_helpful +=1
        elif item["completion"] == "unhelpful":
            num_unhelpful +=1
        else:
            uncategorized.append(item["completion"])
    print(f"num_helpful: {num_helpful}, num_unhelpful: {num_unhelpful}, \n\n uncategorized: {uncategorized}")

if __name__ == "__main__":
    graded_file_path = "./grade_chunks_gpt4o.json"
    prompt_file_path = "./docs_and_prompts_ds.json"
    with open(graded_file_path) as f:
        g_data = json.load(f)
    with open(prompt_file_path) as f:
        p_data = json.load(f)
    for g,p in zip(g_data,p_data):
        if g["prompt"] != p["prompt"]:
            raise Exception("mismatch")
    print_stats_on_data(g_data)
    
