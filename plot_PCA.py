import json
import numpy as np
import chromadb
from chromadb.config import Settings
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import List, Optional
import random
import os

# from torch import chunk

NUMBER_OF_COMPONENTS = 2  # Change to 3 for 3D PCA
NUMBER_OF_CLUSTERS_DISPLAYED = 2  # Limit the number of chunks displayed for clarity
NUMBER_OF_UNTARGETED_CHUNKS_DISPLAYED = 20  # Limit the number of untargeted chunks displayed for clarity
SEED = 42  # For reproducibility
COLLECTION_NAME = "rotated_experiment"
DIRECTORY_ROTATION_DB =  os.path.join(os.getcwd(), "./chroma_rotated_db")

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


# def get_id_targeted_chunk(data, id_triplet: str) -> List[str]:
#         for chunk in data:
#             if chunk["id_triplets"] == id_triplet:
#                 return [f'{chunk_id.split("|")[0]}|{chunk_id[-2]}|{chunk_id[-1]}' for chunk_id in chunk.get("targeted_chunk", [])]

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
        # result = collection.get(
        #     where={
        #         "$and": [
        #             {"triplet_index": {"$eq": triplet_index}},
        #             {"document_id": {"$eq": document_id}},
        #             {"phrase_seq": {"$eq": phrase_seq}},
        #         ]
        #     },
        #     include=["embeddings"],
        # )
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

# Apply PCA and plot
# def plot_pca(
#     embeddings: np.ndarray,
#     labels: List[str],
#     title: str,
#     n_components: int = 2,
#     save_path: Optional[str] = None,
# ):
#     pca = PCA(n_components=n_components)
#     reduced_embeddings = pca.fit_transform(embeddings)

#     plt.figure(figsize=(10, 8))
#     if n_components == 2:
#         plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="viridis", alpha=0.6)
#         plt.xlabel("PCA Component 1")
#         plt.ylabel("PCA Component 2")
#     else:
#         ax = plt.axes(projection="3d")
#         ax.scatter3D(
#             reduced_embeddings[:, 0],
#             reduced_embeddings[:, 1],
#             reduced_embeddings[:, 2],
#             c=labels,
#             cmap="viridis",
#             alpha=0.6,
#         )
#         ax.set_xlabel("PCA Component 1")
#         ax.set_ylabel("PCA Component 2")
#         ax.set_zlabel("PCA Component 3")

#     plt.title(title)
#     plt.colorbar(label="Cluster ID")
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.show()

def plot_pca(
    cluster_embeddings: List[List[np.ndarray]],
    title: str,
    n_components: int = 2,
    save_path: Optional[str] = None,
):
    # Flatten all points and build cluster labels
    all_points = []
    cluster_labels = []
    for cluster_idx, cluster in enumerate(cluster_embeddings):
        for point in cluster:
            all_points.append(point)
            cluster_labels.append(cluster_idx)
    all_points = np.vstack(all_points)  # shape: (N_total, D)
    # PCA reduction
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(all_points)
    
    plt.figure(figsize=(10, 8))
    # Choose colors: One color per cluster
    colors = plt.cm.tab10(cluster_labels) if max(cluster_labels) < 10 else plt.cm.tab20(cluster_labels)
    if n_components == 2:
        plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels, cmap="tab10", alpha=0.6)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
    else:
        ax = plt.axes(projection="3d")
        scatter = ax.scatter3D(
            reduced[:, 0],
            reduced[:, 1],
            reduced[:, 2],
            c=cluster_labels,
            cmap="tab10" if max(cluster_labels) < 10 else "tab20",
            alpha=0.6,
        )
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_zlabel("PCA Component 3")
    plt.title(title)
    # Legend for clusters
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab10(i), markersize=12) 
               for i in range(max(cluster_labels)+1)]
    plt.legend(handles, [f"Cluster {i}" for i in range(max(cluster_labels)+1)], title="Cluster")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# Example usage
if __name__ == "__main__":
    # Load the JSON file with targeted chunks
    with open("documents_RAGBench/merged_id_triplets_with_metadata2.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    list_of_chunk_ids = get_all_chunk_ids(data)
    list_of_targeted_chunk_ids = get_list_id_targeted_chunk(data)
    
    list_of_untargeted_chunk_ids = get_id_untargeted_chunk(data) # List[List[str]]
    list_of_cluster_ids = get_id_clusters(data) # List[str]

    # targeted_chunks = get_id_targeted_chunk(data, "your_query_id_here")  
        
    # Select a query and its targeted chunks
    # query_id = "your_query_id_here"  # Replace with a valid query ID from your JSON
    # targeted_chunks = get_targeted_chunks(query_id)

    list_of_cluster_embeddings = []
    for e in list_of_cluster_ids:
        embedding = fetch_embeddings(
            collection_path=DIRECTORY_ROTATION_DB,
            collection_name=COLLECTION_NAME,
            chunk_ids=[e][0],
            is_rotated=False,
        )
        list_of_cluster_embeddings.append(embedding)

    # Fetch embeddings from the rotated collection
    list_of_targeted_chunk_embeddings = fetch_embeddings(
        collection_path=DIRECTORY_ROTATION_DB,
        collection_name=COLLECTION_NAME,
        chunk_ids=list_of_untargeted_chunk_ids,
        is_rotated=True,
    )

    list_of_cluster_embeddings.append(list_of_targeted_chunk_embeddings)

    # to makes things easier append list_of_targeted_chunk_embeddings to list_of_cluster_embeddings

    # Assign labels (e.g., cluster IDs or targeted vs. non-targeted)
    # labels = np.arange(len(targeted_chunks))  # Simple numeric labels for clusters
    plot_pca(list_of_cluster_embeddings, title="PCA of Clusters", n_components=2)


    # # Plot PCA
    # plot_pca(
    #     embeddings=rotated_embeddings,
    #     labels=labels,
    #     title=f"PCA of Rotated Embeddings for Query {query_id}",
    #     save_path="pca_rotated.png",
    # )