
generate a database

python ingestion_pipeline.py 



# Experiments :

## Experiment : Accessing the effectivness of the rotation to prevent the access 

### Step 1 — collect stable ground truth
python ground_truth_collector.py --dataset test_set.json --runs 5


python ground_truth_collector.py --dataset documents_RAGBench/merged_id_triplets.json --runs 5


### Step 2 — run rotation experiment
python rotation_experiment.py

### Step 3 — plot everything
python rotation_plots.py