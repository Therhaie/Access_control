import json
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Load your JSON data
with open("results/ground_truth_retrievals.json", encoding="utf-8") as fh:
    data = json.load(fh)

store_id = [da["triplet_index"] for da in data]
# print(len(store_id))

dict_store_id = {}
for chunk in data:
    sub_list = []
    for stable_chunk in chunk["stable_chunks"]:
        sub_list.append(stable_chunk["triplet_index"])
    dict_store_id[chunk["triplet_index"]] = sub_list

# print(dict_store_id)

store_id = sorted(store_id)

# Create a dictonary to turn the pseudo document_id into a number
id_to_num = {id: num for num, id in enumerate(store_id)}

matrix = np.zeros((len(store_id), len(store_id)), dtype=int)

for key, value in dict_store_id.items():
    row_index = id_to_num[key]
    for doc_id in value:
        if doc_id in id_to_num:
            col_index = id_to_num[doc_id]
            matrix[row_index][col_index] += 1

# Plot the heatmap
plt.figure(figsize=(12, 8))
# sns.heatmap(
#     matrix,
#     annot=False,  # Disable annotations
#     cmap="YlGnBu",
#     cbar_kws={"label": "Count"},
#     linewidths=0.5,
#     linecolor="lightgray"
# )
sns.heatmap(
    matrix,
    annot=False,               # No number annotations in cells
    cmap="Greys",              # Greyscale color map, white for low
    cbar_kws={"label": "Count"},
    linewidths=0.5,
    linecolor="lightgray",
    square=True,               # Make each cell square
    mask=(matrix == 0)         # Ensure 0-values are white (masked)
)
plt.title("Heatmap of triplet_index vs document_id (Count)")
plt.xlabel("Chunks retrieved as stable")
plt.ylabel("Triplet Index query")
plt.xticks(ticks=np.arange(len(store_id)) + 0.5, labels=store_id, rotation=90)
plt.yticks(ticks=np.arange(len(store_id)) + 0.5, labels=store_id, rotation=0)
plt.tight_layout()
plt.savefig("heatmap_triplet_document_improved.png", dpi=300, bbox_inches="tight")
plt.show()