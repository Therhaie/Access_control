
generate a database

python ingestion_pipeline.py 



# Experiments :

## Experiment : Accessing the effectivness of the rotation to prevent the access 

### Step 1 — collect stable ground truth
python ground_truth_collector.py --dataset documents_RAGBench/merged_id_triplets.json --runs 5

### Step 2 — run rotation experiment
python rotation_experiment.py

### Step 3 — plot everything
python rotation_plots.py

## Experiment : Accessing the efficiency of the rotation and the extra dimension access control methods
### 1. Collect stable ground truth
python ground_truth_collector.py --dataset documents_RAGBench/merged_id_triplets.json --runs 5

### 2. Run rotation experiment (builds rotated Chroma collection + registry)
python rotation_experiment.py

### 3. Run dimension experiment (builds one Chroma collection per config)
python dim_experiment.py

### 4. Latency benchmark (also marks chunks as restricted in original collection)
python latency_benchmark.py

### 5. TRACE across all conditions (requires registry + dim collections + restricted tags)
python trace_conditions_eval.py --dataset test_set.json

### 6. All plots
python experiment_plots.py --plot all