import ollama
from milvus_rag_client import MilvusRagClient


cli = MilvusRagClient(db_path="./m2.db")

query = "谁有糖尿病？"
query_emb = ollama.embeddings(model="bge-m3", prompt=query)["embedding"]

coll_name = "test2"
hits = cli.search(
    collection_name=coll_name,
    query_vectors=[query_emb],
    limit=5,
    output_fields=["id", "text", "source", "seq"],
    filter="source == '1.md'",
)


print("原始检索结果")
print("——" * 50)
if hits and len(hits) > 0 and len(hits[0]) > 0:
    for rank, hit in enumerate(hits[0], 1):
        ent = hit["entity"]
        print(f"  {rank:2}. score={hit['distance']:.4f}")
        print(f"     id    : {hit['id']}")
        print(f"     text  : {ent.get('text','')[:80]}...")
        print(f"     source: {ent.get('source')}")
        print()
else:
    print("没有找到匹配结果")
print("——" * 50)

print("\n格式化检索结果")
print("——" * 50)
docs = cli.find_docs(
    collection_name=coll_name,
    query_vectors=[query_emb],
    limit=5,
    return_fields=["id", "text"],
)
for doc in docs:
    print(doc)
print("——" * 50)
