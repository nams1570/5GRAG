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

ACCEPTABLE_RELEASES = ["17"]
ACCEPTABLE_SERIES = [f"{series}" for series in range(38,39,1)]
ACCEPTABLE_FILENAMES = ["38211-i10","38212-i10","38214-i10","38213-i10","38331-i00","38181-i00","38133-i40","38321-i00","38174-i30","38300-i00","38104-i40","38175-h50","38113-i10","38355-i00","38108-i10","38106-i30","38114-i00"]

def filter_ds_by_metadata(ds:pd.DataFrame,byRelease:bool,bySeries:bool,byFilename:bool):
    if byRelease:
        ds = ds[ds["release"].isin(ACCEPTABLE_RELEASES)]
    if bySeries:
        ds = ds[ds["series"].isin(ACCEPTABLE_SERIES)]
    if byFilename:
        ds = ds[ds["file_name"].isin(ACCEPTABLE_FILENAMES)]
    return ds

argparser = argparse.ArgumentParser()
argparser.add_argument('--output_path', '-o', type=str,default="./ds.json")
argparser.add_argument('--size',type=int,default=100)
argparser.add_argument('--by_release',type=lambda x: x in ["1","True","true","yes"],default=True)
argparser.add_argument('--by_series',type=lambda x: x in ["1","True","true","yes"],default=True)
argparser.add_argument('--by_filename',type=lambda x: x in ["1","True","true","yes"],default=False)

args = argparser.parse_args()
print(f"args are {args} \n")
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
final_ds = filter_ds_by_metadata(merged_ds,args.by_release,args.by_series,args.by_filename)

if len(final_ds) == 0:
    raise Exception("Empty dataset! Check if a filter is too stringent")

results = []
for i in range(min(args.size,len(final_ds))):
    ds_obj = final_ds.iloc[i]
    results.append({'question': ds_obj['Statement'],'ground_truth': ds_obj['Answer'], 'release': ds_obj['release'] ,\
              'series': ds_obj['series'],'file_name': ds_obj['file_name']})

print(f"There are {len(results)} entries in the final dataset")

with open(args.output_path, 'w') as f:
    json.dump(results, f, indent=4)
    
    
