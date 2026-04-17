import csv
import numpy as np
import chromadb

client = chromadb.PersistentClient(path="db/chroma_db")
collection = client.get_collection("langchain")

results = collection.get(include=["embeddings", "documents", "metadatas"])
embeddings = np.array(results["embeddings"])

# vectors.tsv
with open("vectors.tsv", "w", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    for row in embeddings:
        writer.writerow(row)

# metadata.tsv — multi-column with header
with open("metadata.tsv", "w", encoding="utf-8") as f:
    f.write("chunk\tsource\n")
    for doc, meta in zip(results["documents"], results["metadatas"]):
        text = doc.replace("\n", " ").replace("\ufeff", "").replace("\t", " ")
        source = meta.get("source", "unknown").replace("\t", " ")
        f.write(f"{text}\t{source}\n")

print(f"Exported {len(embeddings)} embeddings.")