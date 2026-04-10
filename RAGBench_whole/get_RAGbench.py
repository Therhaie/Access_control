from datasets import load_dataset
import json

# DATASET_NAME = "galileo-ai/ragbench"
# CONFIG_NAME = ["covidqa", "msmarco"] # "msmarco"

# desired_keys = [
#     'id', 
#     'question', 
#     'response', 
#     'documents_sentences'
# ]

# for i, name in enumerate(CONFIG_NAME):
#     dataset_f'{i}' = load_dataset(DATASET_NAME, name, split="train")
#     filtered_data = [
#     {key: item[key] for key in desired_keys} 
#     for item in dataset_{i}
# ]


# # dataset = load_dataset("galileo-ai/ragbench", "covidqa", split="train")
# # print("\n")
# # print(dataset.features)





# with open("RAGBench_whole/filtered_covidqa.json", "w", encoding='utf-8') as f:
#     json.dump(filtered_data, f, ensure_ascii=False, indent=2)


# # merge the 2 dataset together

DATASET_NAME = "galileo-ai/ragbench"
CONFIG_NAMES = ["covidqa", "msmarco"]
desired_keys = [
    'id',
    'question',
    'response',
    'documents_sentences'
]

all_data = []

for config_name in CONFIG_NAMES:
    dataset = load_dataset(DATASET_NAME, config_name, split="train")
    filtered_data = [
        {key: item[key] for key in desired_keys if key in item}
        for item in dataset
    ]
    all_data.extend(filtered_data)

# Now all_data contains all your merged/filtered datapoints
with open("RAGBench_whole/merged_dataset.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

