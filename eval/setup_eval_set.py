from datasets import load_dataset
import pandas as pd
import ast
"""
final_ds
cols: @Statement: the question, @Answer: the gold answer, @id: the tele_data id, @category: always standard \
@content: the text chunk used to make the question, @series: limited to series in ACCEPTABLE_SERIES \
@release: limited to releases in ACCEPTABLE_RELEASES, @file_name: the file where the content chunk was pulled from
"""

ACCEPTABLE_RELEASES = ["17","18"]
ACCEPTABLE_SERIES = [f"{series}" for series in range(31,39,1)]

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
print(len(final_ds))
