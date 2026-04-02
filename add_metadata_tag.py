# """
# add_metadata_tag.py
# ======================
# Add the metadata tag to the file merged_id_triplets_no_duplicates.json based on ground_truth_retrievals.json so every experiment in the future can be done with the same baseline. The output file is merged_id_triplets_no_duplicates_with_metadata.json.

# Pipeline
# ---
# 1. Load the ground truth retrievals from ground_truth_retrievals.json.
# 2. Create a set of unique identifiers for each chunk based on triplet_index, document_id, and phrase_seq to ensure that each chunk is only tagged once.
# 3. Load the merged_id_triplets_no_duplicates.json file.
# 4. Iterate through the ground truth retrievals and for each stable chunk, find the corresponding entry in the merged data and add the chunk identifier as a metadata tag.
# 5. Save the modified merged data to a new file named merged_id_triplets_with_metadata.json.
# """


import json
import os
import time
# Load the ground truth retrievals
with open("results/ground_truth_retrievals.json", encoding="utf-8") as fh:
    ground_truth_data = json.load(fh)

# Get the list of triplet_index to prevent a chunk to have multiple metadata tags

list_identifier = set()
for entries in ground_truth_data:
    for entry in entries["stable_chunks"]:
        list_identifier.add(f'{entry["triplet_index"]}|{entry["document_id"]}{entry["phrase_seq"]}')
        # len(list_identifier) = 993


# Load the merged_id_triplets_no_duplicates.json file
with open("documents_RAGBench/merged_id_triplets_no_duplicates.json", encoding="utf-8") as fh:
    merged_data = json.load(fh)

list_append_chunk = []
t0 = time.time()
while len(list_append_chunk) < len(list_identifier):
    print(f"Processing... {len(list_append_chunk)}/{len(list_identifier)} triplet indices assigned with metadata tags.")
    print(f"Time elapsed: {time.time() - t0:.2f} seconds")
    # Iterate though all query in ground truth
    for entries in ground_truth_data:

        for entry in entries["stable_chunks"]:

            triplet_index = entry.get("triplet_index")
            document_id = entry.get("document_id")
            phrase_seq = entry.get("phrase_seq")

            chunk_identifier = f"{triplet_index}|{document_id}{phrase_seq}"
            if chunk_identifier not in list_append_chunk:
                list_append_chunk.append(chunk_identifier)

                # need to find the query associated in merged_data with the same triplet_index and add chunk_identifier
                for chunk in merged_data:
                    if chunk.get("id_triplets") == triplet_index:
                        
                        if "targeted_chunk" not in chunk:
                            chunk["targeted_chunk"] = []
                        chunk["targeted_chunk"].append(chunk_identifier)
                        

                        break  # Stop after finding the first match
                for doc in merged_data:
                    doc_triplet_index = doc.get("id_triplets")
                    for sentence in doc.get("sentences", []):
                        identifier_sentence = f"{doc_triplet_index}|{sentence[0]}"
                        if chunk_identifier == identifier_sentence:
                            sentence.append(doc.get("id_triplets"))
                            break  # Stop after finding the first 
                    if chunk_identifier == identifier_sentence:
                        break # break the second for loop
                if chunk_identifier == identifier_sentence:
                    break # break this loop to only add the first best chunk for each query through this loop

# After all processing and modifications are complete, save to a new file
with open("documents_RAGBench/merged_id_triplets_with_metadata.json", "w", encoding="utf-8") as out_fh:
    json.dump(merged_data, out_fh, ensure_ascii=False, indent=2)

print("Modifications saved to documents_RAGBench/merged_id_triplets_with_metadata.json")

