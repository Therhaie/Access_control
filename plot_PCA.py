import json
import numpy as np
import chromadb
from chromadb.config import Settings
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import List, Optional
import random
import os
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import umap.umap_ as umap


# from torch import chunk

NUMBER_OF_COMPONENTS = 2  # Change to 3 for 3D PCA
NUMBER_OF_CLUSTERS_DISPLAYED = 3  # Limit the number of chunks displayed for clarity
NUMBER_OF_UNTARGETED_CHUNKS_DISPLAYED = 40  # Limit the number of untargeted chunks displayed for clarity
SEED = 42  # For reproducibility


COLLECTION_NAME_ROTATION = "rotated_experiment"
DIRECTORY_ROTATION_DB =  os.path.join(os.getcwd(), "./chroma_rotated_db")
COLLECTION_NAME_BASELINE = 'baseline_db'
DIRECTORY_BASELINE_DB = os.path.join(os.getcwd(), "./chroma_db")
SAVE_DIR = os.path.join(os.getcwd(), "plots")

# ensure directory for saving plots exists
os.makedirs(SAVE_DIR, exist_ok=True)


def get_all_chunk_ids(data) -> List[str]:
    chunk_ids : List[str] = []
    for chunk in data:
        id_triplet = chunk.get("id_triplets", "")
        for sentence in chunk.get("sentences", []):
            chunk_ids.append(f"{id_triplet}|{sentence[0][-2]}|{sentence[0][-1]}")

    # for sentences in data.get("sentences", []):
    #     id_triplets = sentences.get("id_triplets", "")
    #     for sentence in sentences:
    #         document_id = sentence[-2]
    #         phrase_seq = sentence[-1]
    #         chunk_ids.append(f"{id_triplets}|{document_id}|{phrase_seq}")
    return chunk_ids

def get_list_id_targeted_chunk(data) -> List[str]:
    list_of_chunk_ids = []
    for chunk in data:
        for chunk_id in chunk.get("targeted_chunk", []):
            if f'{chunk_id.split("|")[0]}|{chunk_id[-2]}|{chunk_id[-1]}' not in list_of_chunk_ids:
                list_of_chunk_ids.append(f'{chunk_id.split("|")[0]}|{chunk_id[-2]}|{chunk_id[-1]}')
    return list_of_chunk_ids


# define a function to get a fixed number of untargeted chunks randomly with a seed for reproducibility

def get_id_untargeted_chunk(data) -> List[str]:
    untargeted_chunks = []
    list_of_chunk_ids = get_all_chunk_ids(data)
    list_of_targeted_chunk_ids = get_list_id_targeted_chunk(data)
    for chunk_id in list_of_chunk_ids:
        if chunk_id not in list_of_targeted_chunk_ids:
            untargeted_chunks.append(chunk_id)
    return np.random.choice(untargeted_chunks, size=min(NUMBER_OF_UNTARGETED_CHUNKS_DISPLAYED, len(untargeted_chunks)), replace=False).tolist()

def get_id_clusters(data) :
    """Give the id of NUMBER_OF_CLUSTERS_DISPLAYED targeted chunks cluster using the SEED, """
    clusters = []
    for chunk in data:
        cluster = []
        for targeted_chunk in chunk.get("targeted_chunk", []):
            cluster.append(f'{targeted_chunk.split("|")[0]}|{targeted_chunk[-2]}|{targeted_chunk[-1]}')
        clusters.append(cluster)
    selected_indices = random.sample(range(len(clusters)), NUMBER_OF_CLUSTERS_DISPLAYED)
    return [clusters[i] for i in selected_indices]
            

# Extract targeted chunks (e.g., for a specific query)
def get_targeted_chunks(query_id: str) -> List[str]:
    for record in data:
        if record["id_triplets"] == query_id:
            return record.get("targeted_chunk", [])
    return []

# Fetch embeddings from ChromaDB
def fetch_embeddings(
    collection_path: str,
    collection_name: str,
    chunk_ids: List[str],
    is_rotated: bool = False,
    is_dim: bool = False,
) -> np.ndarray:
    client = chromadb.PersistentClient(
        path=collection_path,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(name=collection_name)

    embeddings = []
    for chunk_id in chunk_ids:
        # Parse chunk_id (format: "triplet_index|document_id|phrase_seq")
        triplet_index = str(chunk_id.split("|")[0])
        document_id = str(chunk_id.split("|")[1])
        phrase_seq = str(chunk_id.split("|")[2])
        
        chroma_id = f"{triplet_index}_{document_id}_{phrase_seq}"
        if collection_path == DIRECTORY_ROTATION_DB :
            result = collection.get(ids=[chroma_id], include=["embeddings"])
        elif collection_path == DIRECTORY_BASELINE_DB:
            result = collection.get(
                where={
                    "$and": [
                        {"triplet_index": {"$eq": triplet_index}},
                        {"document_id":   {"$eq": document_id}},
                        {"phrase_seq":    {"$eq": phrase_seq}},
                    ]
                },
                include=["embeddings"],
            )

        if len(result["embeddings"][0]) > 0:
            embeddings.append(result["embeddings"][0])

    return np.array(embeddings, dtype=np.float32)


def plot_pca(
    cluster_embeddings: List[List[np.ndarray]],
    title: str,
    n_components: int = 2,
    method: str = "pca",
    save_path: Optional[str] = None,
):
    # Flatten points and build labels
    all_points = []
    cluster_labels = []
    for cluster_idx, cluster in enumerate(cluster_embeddings):
        for point in cluster:
            all_points.append(point)
            cluster_labels.append(cluster_idx)
        all_points.append(np.mean(cluster, axis=0))
        cluster_labels.append(NUMBER_OF_CLUSTERS_DISPLAYED + 1)  # Label for cluster center

    all_points = np.vstack(all_points)

        # Dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=SEED)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=SEED)
    else:
        raise ValueError(f"Unknown method: {method}")

    # PCA reduction
    # pca = PCA(n_components=n_components)
    # reduced = pca.fit_transform(all_points)
    reduced = reducer.fit_transform(all_points)

    n_clusters = len(cluster_embeddings)

    # Color map definition
    colors = ["red", "blue", "green", "yellow", "purple", "magenta", "yellow", "brown", "pink"]
    colors_used = colors[:n_clusters-1] + ["gray"] + ["black"] # untargeted cluster in gray and cluster centers in black
    custom_map = ListedColormap(colors_used)  

    plt.figure(figsize=(10, 8))
    # Plot scatter
    if n_components == 2:
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1],
            c=cluster_labels, cmap=custom_map, alpha=0.6
        )
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
    else:
        ax = plt.axes(projection="3d")
        scatter = ax.scatter3D(
            reduced[:, 0], reduced[:, 1], reduced[:, 2],
            c=cluster_labels, cmap=custom_map, alpha=0.6
        )
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_zlabel(f"{method.upper()} Component 3")

    plt.title(title)

    # Legend: Use same colors (by index) as the plotted points
    legend_labels = [f"Cluster {i}" for i in range(n_clusters - 1)] + ["Untargeted"] + ["Cluster Center"]
    handles = [
        plt.Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=custom_map(i), markersize=12, label=legend_labels[i]
        )
        for i in range(NUMBER_OF_CLUSTERS_DISPLAYED + 2)  # range(n_clusters)
    ]

    distance_handle = plt.Line2D(
        [0], [0], color='black', linestyle='--', label='Distance : Cosine Similarity'
    )
    handles.append(distance_handle)
    legend_labels.append('Distance : Cosine Similarity')

    plt.legend(handles, legend_labels, title="Clusters")

    # Access the cluster centers 
    center_label = NUMBER_OF_CLUSTERS_DISPLAYED + 1
    # center_indices = (cluster_labels == center_label)
    center_indices = [i for i, label in enumerate(cluster_labels) if label == center_label]
    cluster_centers_pca = reduced[center_indices]     
    
    for x in cluster_centers_pca:
        x_idx = int(np.where((reduced == x).all(axis=1))[0])  # Find the index of the cluster center in the reduced space
        for y  in cluster_centers_pca:
            y_idx = int(np.where((reduced == y).all(axis=1))[0]) # Find the index of the other cluster center in the reduced space
            if not np.array_equal(x, y):
                plt.plot([x[0], y[0]], [x[1], y[1]], color="black", linestyle="--", alpha=0.5)
                x_embedding = all_points[x_idx]
                y_embedding = all_points[y_idx]
                cos_sim = float(cosine_similarity([x_embedding], [y_embedding]))
                plt.annotate(
                    f"Distance: {cos_sim:.2f}",
                    xy=((x[0] + y[0]) / 2, (x[1] + y[1]) / 2),
                    fontsize=8, color="black", weight="bold"
                )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# Example usage
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    # Load the JSON file with targeted chunks
    with open("documents_RAGBench/merged_id_triplets_with_metadata2.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    list_of_chunk_ids = get_all_chunk_ids(data)
    list_of_targeted_chunk_ids = get_list_id_targeted_chunk(data)
    
    list_of_untargeted_chunk_ids = get_id_untargeted_chunk(data) # List[List[str]]
    list_of_cluster_ids = get_id_clusters(data) # List[str]


    list_of_cluster_embeddings = []
    for e in list_of_cluster_ids:
        embedding = fetch_embeddings(
            collection_path=DIRECTORY_ROTATION_DB,
            collection_name=COLLECTION_NAME_ROTATION,
            chunk_ids=[e][0],
            is_rotated=False,
        )
        list_of_cluster_embeddings.append(embedding)

    # Fetch embeddings from the rotated collection
    list_of_targeted_chunk_embeddings = fetch_embeddings(
        collection_path=DIRECTORY_BASELINE_DB,
        collection_name=COLLECTION_NAME_BASELINE,
        chunk_ids=list_of_untargeted_chunk_ids,
        is_rotated=True,
    )

    list_of_cluster_embeddings.append(list_of_targeted_chunk_embeddings)



    method = "pca"  # Choose between "pca", "umap", or "tsne"
    save_path = os.path.join(SAVE_DIR, f"{method}_clusters.png")
    plot_pca(list_of_cluster_embeddings, title=f"Clusters representation using {method.upper()}, seed : {SEED}", n_components=2, method=method, save_path=save_path)
