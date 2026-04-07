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

# from dim_experiment_split_logged_high_value_encoding import DEFAULT_LARGE_VAL, EXTRA_DIM_CONFIG,augment_query, _l2_norm, ExtraDimConfig
from ingestion_pipeline import get_embedding_model

# from torch import chunk

NUMBER_OF_COMPONENTS = 2  # Change to 3 for 3D PCA
NUMBER_OF_CLUSTERS_DISPLAYED = 3  # Limit the number of chunks displayed for clarity
NUMBER_OF_UNTARGETED_CHUNKS_DISPLAYED = 40  # Limit the number of untargeted chunks displayed for clarity
SEED = 42  # For reproducibility


COLLECTION_NAME_ROTATION = "rotated_experiment"
DIRECTORY_ROTATION_DB =  os.path.join(os.getcwd(), "./chroma_rotated_db")

COLLECTION_NAME_AUGMENTED ="augmented_db_norm_1000000000.0" # low value encode access control # "augmented_db_1000.0" large value encode security #"augmented_db"
DIRECTORY_AUGMENTED_DB = os.path.join(os.getcwd(), "./chroma_aug_db")

COLLECTION_NAME_BASELINE = "baseline_db"
DIRECTORY_BASELINE_DB = os.path.join(os.getcwd(), "./chroma_db")

SAVE_DIR = os.path.join(os.getcwd(), "plots")

# ensure directory for saving plots exists
os.makedirs(SAVE_DIR, exist_ok=True)


def get_all_chunk_ids(data) -> List[str]:
    """Extract all chunk ids from the data, in the format "triplet_index|document_id|phrase_seq"."""
    chunk_ids : List[str] = []
    for chunk in data:
        id_triplet = chunk.get("id_triplets", "")
        for sentence in chunk.get("sentences", []):
            chunk_ids.append(f"{id_triplet}|{sentence[0][-2]}|{sentence[0][-1]}")
    return chunk_ids

def get_list_id_targeted_chunk(data) -> List[str]:
    """Extract all targeted chunk ids from the data, in the format "triplet_index|document_id|phrase_seq"."""
    list_of_chunk_ids = []
    for chunk in data:
        for chunk_id in chunk.get("targeted_chunk", []):
            if f'{chunk_id.split("|")[0]}|{chunk_id[-2]}|{chunk_id[-1]}' not in list_of_chunk_ids:
                list_of_chunk_ids.append(f'{chunk_id.split("|")[0]}|{chunk_id[-2]}|{chunk_id[-1]}')
    return list_of_chunk_ids


# for the embedding
# def _l2_norm(v: np.ndarray) -> np.ndarray:
#     n = np.linalg.norm(v)
#     return v / n if n > 1e-10 else v

# def augment_query(
#     base_vec:    np.ndarray,
#     cfg:         ExtraDimConfig,
#     query_index: int,
#     authorised:  bool,
# ) -> np.ndarray:
#     """
#     Append N extra dims to a query vector.
#     authorised=True  → slot[query_index] = large_value  (aligns with own chunks)
#     authorised=False → all-zero extra dims               (diverges from own chunks)
#     """
#     extra = np.zeros(cfg.n_queries, dtype=np.float32)
#     # extra.fill(DEFAULT_LARGE_VAL)
#     if authorised:
#         # extra[query_index] = cfg.large_value
#         extra[query_index] = DEFAULT_LARGE_VAL

#     aug = np.concatenate([base_vec.astype(np.float32), extra])
#     return _l2_norm(aug) if cfg.normalize_after else aug



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
    questions : list[str] = []
    for chunk in data:
        cluster = []
        questions.append(chunk.get("question", ""))
        for targeted_chunk in chunk.get("targeted_chunk", []):
            cluster.append(f'{targeted_chunk.split("|")[0]}|{targeted_chunk[-2]}|{targeted_chunk[-1]}')
        clusters.append(cluster)
    selected_indices = random.sample(range(len(clusters)), NUMBER_OF_CLUSTERS_DISPLAYED)
    return [clusters[i] for i in selected_indices], [questions[i] for i in selected_indices]
            

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
        if collection_path == DIRECTORY_AUGMENTED_DB:
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


# def plot_pca(
#     cluster_embeddings: List[List[np.ndarray]],
#     title: str,
#     n_components: int = 2,
#     method: str = "pca",
#     save_path: Optional[str] = None,
#     plot_centers: bool = True,
#     embed_questions: Optional[List[np.ndarray]] = None

# ):
#     # Flatten points and build labels
#     all_points = []
#     cluster_labels = []
#     for cluster_idx, cluster in enumerate(cluster_embeddings):
#         for point in cluster:
#             all_points.append(point)
#             cluster_labels.append(cluster_idx)
#         if plot_centers:
#             all_points.append(np.mean(cluster, axis=0))
#             cluster_labels.append(NUMBER_OF_CLUSTERS_DISPLAYED + 1)  # Label for cluster center

#     all_points = np.vstack(all_points)

#         # Dimensionality reduction
#     if method == "pca":
#         reducer = PCA(n_components=n_components)
#     elif method == "umap":
#         reducer = umap.UMAP(n_components=n_components, random_state=SEED)
#     elif method == "tsne":
#         reducer = TSNE(n_components=n_components, random_state=SEED)
#     else:
#         raise ValueError(f"Unknown method: {method}")

#     reduced = reducer.fit_transform(all_points)

#     n_clusters = len(cluster_embeddings)

#     # Color map definition
#     colors = ["red", "blue", "green", "yellow", "purple", "magenta", "yellow", "brown", "pink"]
#     colors_used = colors[:n_clusters-1] + ["gray"] # untargeted cluster in gray and cluster centers in black
#     if plot_centers:
#         colors_used =  colors_used + ["black"] # cluster centers in black
#     custom_map = ListedColormap(colors_used)  

#     plt.figure(figsize=(10, 8))
#     # Plot scatter
#     if n_components == 2:
#         scatter = plt.scatter(
#             reduced[:, 0], reduced[:, 1],
#             c=cluster_labels, cmap=custom_map, alpha=0.6
#         )
#         plt.xlabel(f"{method.upper()} Component 1")
#         plt.ylabel(f"{method.upper()} Component 2")
#     else:
#         ax = plt.axes(projection="3d")
#         scatter = ax.scatter3D(
#             reduced[:, 0], reduced[:, 1], reduced[:, 2],
#             c=cluster_labels, cmap=custom_map, alpha=0.6
#         )
#         ax.set_xlabel(f"{method.upper()} Component 1")
#         ax.set_ylabel(f"{method.upper()} Component 2")
#         ax.set_zlabel(f"{method.upper()} Component 3")

#     plt.title(title)
    

#     # Legend: Use same colors (by index) as the plotted points
#     if plot_centers:
#         legend_labels = [f"Cluster {i}" for i in range(n_clusters - 1)] + ["Untargeted"] + ["Cluster Center"]
#     else:
#         legend_labels = [f"Cluster {i}" for i in range(n_clusters - 1)] + ["Untargeted"]


#     handles = [
#         plt.Line2D(
#             [0], [0], marker='o', color='w',
#             markerfacecolor=custom_map(i), markersize=12, label=legend_labels[i]
#         )
#         for i in range(NUMBER_OF_CLUSTERS_DISPLAYED + 2)  # range(n_clusters)
#     ]

#     if plot_centers:
#         distance_handle = plt.Line2D(
#             [0], [0], color='black', linestyle='--', label='Distance : Cosine Similarity'
#         )
#         handles.append(distance_handle)
#         legend_labels.append('Distance : Cosine Similarity')


#     # Access the cluster centers 
#     center_label = NUMBER_OF_CLUSTERS_DISPLAYED + 1
#     # center_indices = (cluster_labels == center_label)
#     center_indices = [i for i, label in enumerate(cluster_labels) if label == center_label]
#     cluster_centers_reduced = reduced[center_indices]     

#     # new loop because the value of x and y are too simuilar leading to multiple indice corresponding to the same value within a cluster
#     for x, x_id in zip(cluster_centers_reduced,center_indices):
#         for y, y_id in zip(cluster_centers_reduced, center_indices):
#             if not np.array_equal(x, y):
#                 plt.plot([x[0], y[0]], [x[1], y[1]], color="grey", linestyle="--", alpha=0.5)
#                 x_embedding = all_points[x_id]
#                 y_embedding = all_points[y_id]
#                 cos_sim = float(cosine_similarity([x_embedding], [y_embedding]))
#                 plt.annotate(
#                     f"Distance: {cos_sim:.9e}",
#                     xy=((x[0] + y[0]) / 2, (x[1] + y[1]) / 2),
#                     fontsize=8, color="black", weight="bold"
#                 )
    
#     plt.legend(handles, legend_labels, title="Clusters")

#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.show()

def plot_pca(
    cluster_embeddings: List[List[np.ndarray]],
    title: str,
    n_components: int = 2,
    method: str = "pca",
    save_path: Optional[str] = None,
    plot_centers: bool = True,
    embed_questions: Optional[List[np.ndarray]] = None,
):
    # Flatten points and build labels
    all_points = []
    cluster_labels = []
    for cluster_idx, cluster in enumerate(cluster_embeddings):
        for point in cluster:
            all_points.append(point)
            cluster_labels.append(cluster_idx)
        if plot_centers:
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

    reduced = reducer.fit_transform(all_points)

    # Reduce embed_questions if provided
    if embed_questions is not None:
        embed_questions = np.vstack(embed_questions)
        reduced_questions = reducer.transform(embed_questions)

    n_clusters = len(cluster_embeddings)

    # Color map definition
    colors = ["red", "blue", "green", "yellow", "purple", "magenta", "yellow", "brown", "pink"]
    colors_used = colors[:n_clusters-1] + ["gray"]  # untargeted cluster in gray
    if plot_centers:
        colors_used = colors_used + ["black"]  # cluster centers in black
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

        # Plot embed_questions as white crosses
        if embed_questions is not None:
            plt.scatter(
                reduced_questions[:, 0], reduced_questions[:, 1],
                c="white", marker="x", s=100, linewidths=2, label="Questions"
            )
    else:
        ax = plt.axes(projection="3d")
        scatter = ax.scatter3D(
            reduced[:, 0], reduced[:, 1], reduced[:, 2],
            c=cluster_labels, cmap=custom_map, alpha=0.6
        )
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_zlabel(f"{method.upper()} Component 3")

        # Plot embed_questions as white crosses (3D)
        if embed_questions is not None:
            ax.scatter3D(
                reduced_questions[:, 0], reduced_questions[:, 1], reduced_questions[:, 2],
                c="white", marker="x", s=100, linewidths=2, label="Questions"
            )

    plt.title(title)

    # Legend: Use same colors (by index) as the plotted points
    if plot_centers:
        legend_labels = [f"Cluster {i}" for i in range(n_clusters - 1)] + ["Untargeted"] + ["Cluster Center"]
    else:
        legend_labels = [f"Cluster {i}" for i in range(n_clusters - 1)] + ["Untargeted"]

    handles = [
        plt.Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=custom_map(i), markersize=12, label=legend_labels[i]
        )
        for i in range(NUMBER_OF_CLUSTERS_DISPLAYED + 2)
    ]

    if embed_questions is not None:
        handles.append(plt.Line2D([0], [0], marker='x', color='white', markersize=12, label="Questions"))

    if plot_centers:
        distance_handle = plt.Line2D(
            [0], [0], color='black', linestyle='--', label='Distance : Cosine Similarity'
        )
        handles.append(distance_handle)
        legend_labels.append('Distance : Cosine Similarity')

    # Access the cluster centers
    center_label = NUMBER_OF_CLUSTERS_DISPLAYED + 1
    center_indices = [i for i, label in enumerate(cluster_labels) if label == center_label]
    cluster_centers_reduced = reduced[center_indices]

    # Plot distances between cluster centers
    for x, x_id in zip(cluster_centers_reduced, center_indices):
        for y, y_id in zip(cluster_centers_reduced, center_indices):
            if not np.array_equal(x, y):
                if n_components == 2:
                    plt.plot([x[0], y[0]], [x[1], y[1]], color="grey", linestyle="--", alpha=0.5)
                    x_embedding = all_points[x_id]
                    y_embedding = all_points[y_id]
                    cos_sim = float(cosine_similarity([x_embedding], [y_embedding]))
                    plt.annotate(
                        f"Distance: {cos_sim:.9e}",
                        xy=((x[0] + y[0]) / 2, (x[1] + y[1]) / 2),
                        fontsize=8, color="black", weight="bold"
                    )
                else:
                    ax.plot([x[0], y[0]], [x[1], y[1]], [x[2], y[2]], color="grey", linestyle="--", alpha=0.5)
                    x_embedding = all_points[x_id]
                    y_embedding = all_points[y_id]
                    cos_sim = float(cosine_similarity([x_embedding], [y_embedding]))
                    ax.text(
                        (x[0] + y[0]) / 2, (x[1] + y[1]) / 2, (x[2] + y[2]) / 2,
                        f"Distance: {cos_sim:.9e}",
                        fontsize=8, color="black", weight="bold"
                    )

    plt.legend(handles=handles, labels=legend_labels, title="Clusters")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# main functions

def plot_rotation():
    random.seed(SEED)
    np.random.seed(SEED)
    # Load the JSON file with targeted chunks
    with open("documents_RAGBench/merged_id_triplets_with_metadata2.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    list_of_chunk_ids = get_all_chunk_ids(data)
    list_of_targeted_chunk_ids = get_list_id_targeted_chunk(data)
    
    list_of_untargeted_chunk_ids = get_id_untargeted_chunk(data) # List[List[str]]
    list_of_cluster_ids, questions = get_id_clusters(data) # List[str]


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
    
    method = "umap"  # Choose between "pca", "umap", or "tsne"
    save_path = os.path.join(SAVE_DIR, f"{method}_clusters.png")
    plot_pca(list_of_cluster_embeddings, title=f"Clusters representation using {method.upper()}, seed : {SEED}", n_components=2, method=method, save_path=save_path)

    method = "tsne"  # Choose between "pca", "umap", or "tsne"
    save_path = os.path.join(SAVE_DIR, f"{method}_clusters.png")
    plot_pca(list_of_cluster_embeddings, title=f"Clusters representation using t-SNE, seed : {SEED}", n_components=2, method=method, save_path=save_path)


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
    list_of_cluster_ids, questions = get_id_clusters(data) # List[str]

    # Get the question
    # embedder = get_embedding_model()
    # embed_questions = []
    # BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    # for question in questions:
    #         raw_q = np.array(
    #             embedder.embed_query(BGE_QUERY_PREFIX + question), dtype=np.float32
    #         )
    #         embed_questions.append(augment_query(raw_q, cfg=EXTRA_DIM_CONFIG, query_index=questions.index(question), authorised=True))


    list_of_cluster_embeddings = []
    for e in list_of_cluster_ids:
        embedding = fetch_embeddings(
            collection_path=DIRECTORY_AUGMENTED_DB,
            collection_name=COLLECTION_NAME_AUGMENTED,
            chunk_ids=[e][0],
            is_rotated=False,
        )
        list_of_cluster_embeddings.append(embedding)

    # Fetch embeddings from the rotated collection
    list_of_targeted_chunk_embeddings = fetch_embeddings(
        collection_path=DIRECTORY_AUGMENTED_DB,
        collection_name=COLLECTION_NAME_AUGMENTED,
        chunk_ids=list_of_untargeted_chunk_ids,
        is_rotated=False,
    )

    list_of_cluster_embeddings.append(list_of_targeted_chunk_embeddings)


    # Plot the query's embedding 


    method = "pca"  # Choose between "pca", "umap", or "tsne"
    save_path = os.path.join(SAVE_DIR, f"{method}_clusters_extra_dim_high_value_access.png")
    plot_pca(list_of_cluster_embeddings, title=f"Clusters representation using {method.upper()}, extra dim, seed : {SEED}", n_components=2, method=method, save_path=save_path, plot_centers=True, embed_questions=embed_questions)
    
    method = "umap"  # Choose between "pca", "umap", or "tsne"
    save_path = os.path.join(SAVE_DIR, f"{method}_clusters_extra_dim_high_value_access.png")
    plot_pca(list_of_cluster_embeddings, title=f"Clusters representation using {method.upper()}, extra dim, seed : {SEED}", n_components=2, method=method, save_path=save_path, plot_centers=True)

    method = "tsne"  # Choose between "pca", "umap", or "tsne"
    save_path = os.path.join(SAVE_DIR, f"{method}_clusters_extra_dim_high_value_access.png")
    plot_pca(list_of_cluster_embeddings, title=f"Clusters representation using t-SNE, extra dim, seed : {SEED}", n_components=2, method=method, save_path=save_path, plot_centers=True)

    

    
