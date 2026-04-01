import chromadb
from config import *
client = chromadb.Client()

collection_name = COLLECTION
client = chromadb.PersistentClient(
        path=CHROMA_PATH,
    )
# try:
#     collection = client.get_collection(collection_name)
#     print(f"Collection '{collection_name}' exists.")
# except ValueError:
#     print(f"Collection '{collection_name}' does not exist.")

print(f"List of collections: {client.list_collections()}")
# client.delete_collection("baseline_db")