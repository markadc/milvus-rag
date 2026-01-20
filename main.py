import random
from llm import chat
from milvus_rag_client import MilvusRagClient
import ollama

cli = MilvusRagClient(db_path="./m2.db")


def main():
    with open("./questions.txt", "r") as f:
        lines = f.readlines()
        random.shuffle(lines)

    for i, line in enumerate(lines, start=1):
        q = line.split(".")[-1].strip()
        query = q
        query_emb = ollama.embeddings(model="bge-m3", prompt=query)["embedding"]
        docs = cli.find_docs(
            collection_name="test2",
            query_vectors=[query_emb],
            limit=5,
            return_fields=["text"],
        )

        # 构建上下文「提示词」
        context = "\n".join([doc["text"] for doc in docs])
        prompt = f"""\
        请你基于提供的文档回答问题
        - 注意1：你只能基于文档回答，如果文档找不到答案就说「我不清楚」；
        - 注意2：回答要简洁；
        - 注意3：回复格式为，具体的回答 + 推断理由。比如我问"小明是谁？"，你回答："小白的哥哥。推断理由：...."

        文档：{context}
        问题：{query}
        """

        # 开始问答
        print(f"Q: {query}")
        print("A: ", end="")
        chat(query, prompt)
        print()
        user_input = input(f"第 {i} 个问题结束（回车继续，q 退出）\n")
        if user_input.strip() == "q":
            print("👋 拜拜！")
            break


if __name__ == "__main__":
    main()
