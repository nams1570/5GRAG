# Info
This subdirectory will contain the code for the baseline models we use to benchmark deepspecs.

# Plan
## 1. Evolution Benchmarking:
We want to show that our retrieval for discussion docs is an improvement over pure semantic retrieval. 
To do so, we want to make sure our baseline has access to the same data that deepspecs does. 
We also want to make sure that the model they use as a base is the same, so we will tie it to the `settings.yml` model_name.
