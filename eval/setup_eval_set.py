from datasets import load_dataset
import pandas as pd
import ast
import json
import argparse
"""
final_ds
cols: @Statement: the question formed by an llm from the content, @Answer: the gold answer, \
@id: the tele_data id, @category: always standard \
@content: the text chunk used to make the question, @series: limited to series in ACCEPTABLE_SERIES \
@release: limited to releases in ACCEPTABLE_RELEASES, @file_name: the file where the content chunk was pulled from
"""
"""
results
keys: @question: the question ,@ground_truth: the gold answer, 
@release: limited to releases in ACCEPTABLE_RELEASES ,@series: limited to series in ACCEPTABLE_SERIES, \
    @file_name: the file where the content chunk was pulled from
"""

ACCEPTABLE_RELEASES = ["17","18"]
ACCEPTABLE_SERIES = [f"{series}" for series in range(31,39,1)]

argparser = argparse.ArgumentParser()
argparser.add_argument('--output_path', '-o', type=str,default="./ds.json")
argparser.add_argument('--size',type=int,default=100)
args = argparser.parse_args()

tele_data = pd.DataFrame(load_dataset("AliMaatouk/Tele-Data",name="standard")["train"])
tele_eval = pd.DataFrame(load_dataset("AliMaatouk/Tele-Eval")["data"])
tele_eval = tele_eval[tele_eval["id"].str.contains("standard",na=False)]

merged_ds = pd.merge(tele_eval,tele_data,on="id")
#turn metadata into 3 separate columns
#firstly we need to convert the metadata column from str into a dictionary
merged_ds["metadata"] = merged_ds["metadata"].map(lambda d: ast.literal_eval(d))
#then we unroll the metadata column into separate columns for "series", "release", and "filename"
merged_ds = pd.concat([merged_ds.drop(["metadata"],axis=1),merged_ds["metadata"].apply(pd.Series)],axis=1)

#filter by releases and versions
final_ds = merged_ds[(merged_ds["release"].isin(ACCEPTABLE_RELEASES)) & (merged_ds["series"]).isin(ACCEPTABLE_SERIES)]

results = []
for i in range(args.size):
    ds_obj = final_ds.iloc[i]
    results.append({'question': ds_obj['Statement'],'ground_truth': ds_obj['Answer'], 'release': ds_obj['release'] ,\
              'series': ds_obj['series'],'file_name': ds_obj['file_name']})

with open(args.output_path, 'w') as f:
    json.dump(results, f, indent=4)
    
    
