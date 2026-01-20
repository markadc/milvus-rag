import os
import ollama
from milvus_rag_client import MilvusRagClient


lines = []
for root, dirnames, filenames in os.walk("./data"):
    for name in filenames:
        file_path = os.path.join(root, name)
        with open(file_path, "r") as f:
            lines.extend(f.readlines())

data = []
for i, line in enumerate(lines, start=1):
    line = line.strip()
    line_emb = ollama.embeddings(model="bge-m3", prompt=line)["embedding"]
    data.append(
        {
            "id": f"test_{i:03d}",
            "vector": line_emb,
            "text": line,
            "source": "2.md" if i % 2 == 0 else "1.md",
            "seq": i,
        }
    )

cli = MilvusRagClient(db_path="./m2.db")

coll_name = "test2"
dim = 1024
cli.create_collection(
    collection_name=coll_name,
    dim=dim,
    metric_type="COSINE",
    auto_id=False,
    drop_if_exist=True,
)

cli.create_index(
    collection_name=coll_name,
    index_type="IVF_FLAT",
    metric_type="COSINE",
    nlist=64,
)

cli.insert(coll_name, data)
