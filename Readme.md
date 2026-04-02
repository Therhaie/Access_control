
generate a database

python ingestion_pipeline.py 

Ports used for the different LLM :
Main            : 8000
Embedder        : 8001
TRACE evaluator : 8002


in dataset available data :
'sentences', 'question', 'response', 'id_triplets'


metadata available from a query in the database :
'triplet_index', 'document_id', 'phrase_seq'


retrieve function return n candidates with 
content, source (=triplet_index), page (=document_id), phrase_seq, bge_score (=similarity), rerank_score if 





collect_chunks for each runs the dictionnary seen append the similarity associated to a "key", 3 data necessary to identify a chunk in the str format *(source (=triplet_index), page (=document_id), phrase_seq)*
then the "key" is also used to store the content of the chunk inside another dictionnary.


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
python trace_conditions_eval.py --dataset documents_RAGBench/merged_id_triplets.json

### 6. All plots
python experiment_plots.py --plot all