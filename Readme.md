
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


retrieving 20 chunks for the ground truth lead to a total of ~ 993 retrieve

Scenario 1 :
    Dataset used
    - all chunk available
    - all desired 'reached_chunk' are blocked
    - all 'target_chunk' are blocked
    
    Metrics computed
    - time to retrieve
    -   include the rotation time of query and of adding dimensions, 

    Methods for blocking
    - rotation
    - add dimension
    - chroma methods or metadata filtering person

remarques : 
test scénario avec tous les documents bloqués
calculer la distance moyenne intercluster pour s'assurer qu'on ne puisse pas reach des documents interdis meme si pas assez de documents à retrieve
implémenter a minima pour le déterminer mais pas forcément le tester

- plot pour afficher les clusters, afficher la distance entre les clusters, PCA (rotation pas de souci, add dimpension à voir)
utiliser les 'targeted_chunk' 
non targeted chunk 0 sur toutes les dimensions devraient se retrouver au milieu 
[dans la partie résultats] nos méthodes produisent ces effets
- tableau : overlap pour valider que les approches fonctions
[ la sécurité est garantie ]
- tableau : performance 
[ comparaison des perfomances dans les résultats ]


Scenario 2 :



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