import json
import csv

'''LLM_predict requires the keys "question" and "ground_truth"'''

#Yifei's questions
def parse_yifei(filepath:str):
    """These are in csv. Keys are "Question", "Best Answer", "First Post Time (YYYY/MM)", "Link of the Post"."""
    results = []
    with open(filepath,"r",encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            st_obj = {'question':row["Question"],'ground_truth':row["Best Answer"],'timestamp':row["First Post Time (YYYY/MM)"],\
                      "source":row["Link of the Post"], "author":"Yifei"}
            results.append(st_obj)
    return results

#Ziyue's questions
def parse_ziyue(filepath:str):
    """These are in json. Keys "Question", "Best Answer" """
    org_objects = []
    results = []
    with open(filepath,"r",encoding="utf-8") as f:
        org_objects = json.load(f)
    for org_obj in org_objects:
        st_obj = {'question':org_obj["Question"],'ground_truth':org_obj["Best Answer"],"timestamp":org_obj["First Post Time (YYYY/MM)"],\
                  "source":org_obj["Link of the Post"],"author":"Ziyue"}
        results.append(st_obj)
    return results

#Aman's questions
def parse_aman(filepath:str):
    """These are in csv. Keys: "Question", "Answer", "Source", "Last Activity"."""
    results = []
    with open(filepath,"r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            st_obj = {'question':row["Question"],'ground_truth':row["Answer"],'timestamp':row["Last Activity"],\
                      "source":row["Source"],"author":"Aman"}
            results.append(st_obj)
    return results

if __name__ == "__main__":
    ziyue_filepath = "Telecomhall_QA_clean.json"
    aman_filepath = "TelecomHall Dataset Aman - Sheet1.csv"
    yifei_filepath = 'TelecomHall_2025_04_07 - Sheet1.csv'
    output_path = 'human_created_telecomhall_questions.json'

    qset1 = parse_ziyue(ziyue_filepath)
    qset2 = parse_aman(aman_filepath)
    qset3 = parse_yifei(yifei_filepath)
    results = qset1 + qset2 + qset3

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)