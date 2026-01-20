from loguru import logger
from pymilvus import MilvusClient, DataType


class MilvusRagClient:
    """
    - Milvus Lite 本地单文件版封装（文件路径如 ./milvus.db）
    - 专为本地 RAG + ollama bge-m3 设计
    - 使用 pymilvus 最新推荐写法（create_schema + prepare_index_params）
    """

    def __init__(self, db_path="./milvus.db"):
        """初始化 Milvus Lite 客户端"""
        self.client = MilvusClient(uri=db_path)
        logger.debug(f"🎉 Milvus 已连接到 {db_path}")

    def list_collections(self):
        """列出当前数据库中所有集合名称"""
        return self.client.list_collections()

    def has_collection(self, collection_name: str) -> bool:
        """判断指定集合是否存在"""
        return self.client.has_collection(collection_name)

    def drop_collection(self, collection_name: str):
        """删除指定集合（谨慎使用）"""
        if self.has_collection(collection_name):
            self.client.drop_collection(collection_name)
            logger.debug(f"已删除集合: {collection_name}")
        else:
            logger.warning(f"集合 {collection_name} 不存在，无需删除")

    def create_collection(
        self,
        collection_name: str,
        dim: int = 1024,
        metric_type: str = "IP",
        auto_id: bool = False,
        enable_dynamic: bool = True,
        drop_if_exist: bool = False,
    ) -> bool:
        """
        创建集合（使用现代 CollectionSchema 写法）
        Args:
            - auto_id=False         自己提供字符串 id（推荐用于 RAG，便于追踪来源）
            - drop_if_exist=True    开发阶段方便重置
        """
        if self.has_collection(collection_name):
            if not drop_if_exist:
                logger.warning(f"集合 {collection_name} 已存在，跳过创建")
                return False
            self.drop_collection(collection_name)

        # 创建 schema
        schema = self.client.create_schema(
            auto_id=auto_id,
            enable_dynamic_field=enable_dynamic,
        )

        # 主键（字符串 id，长度足够大）
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=65535,
        )

        # 向量字段（bge-m3 默认 1024 维）
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim,
        )

        # 创建集合
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
        )

        logger.debug(f"集合创建成功: {collection_name}  dim={dim}  metric={metric_type}")
        return True

    def create_index(
        self,
        collection_name: str,
        index_type: str = "IVF_FLAT",
        metric_type: str = "IP",
        nlist: int = 128,
        index_name: str = "vector_idx",
    ):
        """
        创建向量索引（使用 prepare_index_params 写法，兼容新版 pymilvus）
        常用参数：
            index_type: "IVF_FLAT", "HNSW", "FLAT"（精确搜索）
            nlist: IVF 系列的聚类数，小数据建议 64~256
        """
        if not self.has_collection(collection_name):
            raise ValueError(f"集合 {collection_name} 不存在，无法创建索引")

        index_params = self.client.prepare_index_params()

        # 根据 index_type 自动设置 params
        if "IVF" in index_type:
            params = {"nlist": nlist}
        elif index_type == "HNSW":
            params = {"M": 16, "efConstruction": 200}
        else:
            params = {}  # FLAT 等不需要额外参数

        index_params.add_index(
            field_name="vector",
            index_type=index_type,
            metric_type=metric_type,
            params=params,
            index_name=index_name,
        )

        self.client.create_index(
            collection_name=collection_name,
            index_params=index_params,
        )

        logger.debug(f"索引创建完成: {collection_name}.vector → " f"{index_type} / {metric_type} (nlist={nlist if 'IVF' in index_type else 'N/A'})")

    def insert(
        self,
        collection_name: str,
        data: list[dict],
        batch_size: int = 1000,
    ) -> dict:
        """
        插入数据（支持动态字段）
        data 示例：
        [
            {
                "id": "doc_001_chunk_03",
                "vector": [0.12, -0.34, ..., 0.56],  # 1024维 float list
                "text": "段落原文...",
                "file_name": "2025合同.pdf",
                "chunk_idx": 3,
                "create_time": 1737288000
            },
            ...
        ]
        """
        if not data:
            logger.warning("插入数据为空")
            return {"insert_count": 0}

        res = self.client.insert(
            collection_name=collection_name,
            data=data,
            batch_size=batch_size,
        )

        logger.success(f"插入完成: {res['insert_count']} 条 → {collection_name}")
        return res

    def search(
        self,
        collection_name: str,
        query_vectors: list[list[float]],
        limit: int = 8,
        filter: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list:
        """
        向量搜索
        query_vectors: [[...], [...]] 支持批量查询
        filter: "file_name like '2025%' and chunk_idx < 100"
        """
        if output_fields is None:
            output_fields = ["*"]

        results = self.client.search(
            collection_name=collection_name,
            data=query_vectors,
            limit=limit,
            filter=filter,
            output_fields=output_fields,
        )

        return results

    def find_docs(
        self,
        collection_name: str,
        query_vectors: list[list[float]],
        limit: int = 5,
        output_fields: list[str] | None = None,
        return_fields: list[str] | None = None,
        filter: str | None = None,
    ) -> list:
        """
        查找文档
        Args:
            collection_name: 集合名称
            query_vectors: 查询向量
            limit: 返回的文档数量
            output_fields: 检索返回的字段
            return_fields: 实际返回的字段
            filter: 过滤条件
        Returns:
            文档列表
        """
        hits = self.search(
            collection_name=collection_name,
            query_vectors=query_vectors,
            limit=limit,
            output_fields=output_fields,
            filter=filter,
        )

        if hits and len(hits) > 0 and len(hits[0]) > 0:
            docs = []
            for hit in hits[0]:
                ent = hit["entity"]
                if return_fields is None:
                    doc = ent
                else:
                    doc = {field: ent.get(field) for field in return_fields if field in ent}
                docs.append(doc)
            return docs
        else:
            return []


if __name__ == "__main__":
    import ollama

    client = MilvusRagClient(db_path="./milvus.db")
    COLLECTION = "test1"
    DIM = 1024

    # 1. 创建集合（开发时可强制重建）
    client.create_collection(
        collection_name=COLLECTION,
        dim=DIM,
        metric_type="COSINE",
        auto_id=False,
        drop_if_exist=True,
    )

    # 2. 创建索引
    client.create_index(
        collection_name=COLLECTION,
        metric_type="COSINE",
    )

    # 3. 准备插入数据
    sample_texts = [
        "香港维港夜景在节假日会有烟花表演，非常浪漫。",
        "2026 年大模型在本地部署的成本已经大幅下降。",
        "bge-m3 支持多语言、长文本和稀疏向量检索。",
        "Milvus Lite 非常适合个人电脑跑小型知识库。",
        "全球顶尖大模型有哪些？",
        "你用的什么模型？",
    ]

    insert_data = []
    for idx, txt in enumerate(sample_texts):
        emb = ollama.embeddings(model="bge-m3", prompt=txt)["embedding"]

        insert_data.append(
            {
                "id": f"demo_{idx:03d}",
                "vector": emb,
                "text": txt,
                "source": "2.md" if idx % 2 == 0 else "1.md",
                "seq": idx,
            }
        )

    # 4. 插入
    client.insert(COLLECTION, insert_data)

    # 5. 搜索示例
    query = "大模型"
    query_emb = ollama.embeddings(model="bge-m3", prompt=query)["embedding"]

    hits = client.search(
        collection_name=COLLECTION,
        query_vectors=[query_emb],
        limit=5,
        output_fields=["id", "text", "source", "seq"],
        filter="source == '1.md'",
    )

    print("\n=== 搜索结果 ===")
    if hits and len(hits) > 0 and len(hits[0]) > 0:
        for rank, hit in enumerate(hits[0], 1):
            ent = hit["entity"]
            print(f"  {rank:2}. score={hit['distance']:.4f}")
            print(f"     id    : {hit['id']}")
            print(f"     text  : {ent.get('text','')[:80]}...")
            print(f"     source: {ent.get('source','N/A')}")
            print()
    else:
        print("没有找到匹配结果")
