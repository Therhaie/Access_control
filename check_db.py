import chromadb
from config import *
client = chromadb.Client()

collection_name = COLLECTION
client = chromadb.PersistentClient(
        path="chroma_dim_db",
    )
client = chromadb.PersistentClient(
        path="chroma_aug_db",
    )

# chroma_aug_db
# client = chromadb.PersistentClient(
#         path="chroma_rotated_db",
#     )
    
# try:
#     collection = client.get_collection(collection_name)
#     print(f"Collection '{collection_name}' exists.")
# except ValueError:
#     print(f"Collection '{collection_name}' does not exist.")

print(f"List of collections: {client.list_collections()}")
# client.delete_collection("baseline_db")

# List of collections: [Collection(name=dim_d4_w1.00_1.00_1.00_1.00_nb1_na0), Collection(name=dim_d4_w1.00_1.00_1.00_1.00_nb1_na1), Collection(name=dim_d4_w0.50_0.50_0.50_0.50_nb0_na0), Collection(name=dim_d8_w1.00_1.00_1.00_1.00_1.00_1.00_1.00_1.00_nb0_na0), Collection(name=dim_d4_w1.00_1.00_1.00_1.00_nb0_na0), Collection(name=dim_d4_w1.00_1.00_1.00_1.00_nb0_na1), Collection(name=dim_d4_w2.00_2.00_2.00_2.00_nb0_na0)]
