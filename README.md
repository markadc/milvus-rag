# RAG-Milvus: 本地化检索增强生成系统

基于 Milvus Lite + Ollama + OpenAI API 的本地化 RAG (Retrieval-Augmented Generation) 系统，使用现代化的 Python 包管理器 uv 进行依赖管理。

## 🚀 项目特性

- **本地化部署**: 使用 Milvus Lite 单文件数据库，无需外部服务
- **高效嵌入**: 集成 Ollama bge-m3 模型进行文本嵌入
- **灵活生成**: 支持 OpenAI 兼容 API，可对接多种 LLM
- **现代化管理**: 使用 uv 包管理器，提供快速依赖安装和虚拟环境管理
- **完整 RAG 流程**: 从数据摄入到检索生成的全链路实现

## 📋 系统要求

- Python >= 3.12
- Ollama (用于本地嵌入模型)
- uv (Python 包管理器)

## 🛠️ 环境初始化

### 1. 安装 uv

```bash
# macOS
brew install uv

# 或者使用 pip 安装
pip install uv
```

### 2. 克隆项目并进入目录

```bash
git clone <your-repo-url>
cd rag-milvus
```

### 3. 安装依赖

```bash
# uv 会自动创建虚拟环境并安装所有依赖
uv sync
```

### 4. 安装 Ollama 并下载模型

```bash
# 安装 Ollama
brew install ollama

# 启动 Ollama 服务
ollama serve

# 下载 bge-m3 嵌入模型
ollama pull bge-m3
```

### 5. 配置环境变量

创建 `.env` 文件并配置 OpenAI API：

```bash
# .env
API_KEY=your_openai_api_key
BASE_URL=https://api.openai.com/v1  # 或其他兼容 API 地址
MODEL=gpt-3.5-turbo  # 或其他模型名称
```

## 📊 使用流程

### 1. 数据摄入 (1_add_data_to_coll.py)

将文档数据向量化并存储到 Milvus 数据库：

```bash
uv run 1_add_data_to_coll.py
```

该脚本会：

- 读取 `data/` 目录下的所有文档
- 使用 bge-m3 模型生成文本嵌入向量
- 将向量和原始文本存储到 Milvus 集合中

### 2. 向量检索 (2_retrieval.py)

测试向量检索功能：

```bash
uv 2_retrieval.py
```

该脚本演示了如何：

- 将查询文本转换为向量
- 在 Milvus 中搜索相似文档
- 按相似度排序返回结果

### 3. RAG 查询 (3_query.py)

完整的检索增强生成流程：

```bash
uv run 3_query.py
```

该脚本展示了：

- 将用户查询向量化
- 检索相关文档片段
- 构建上下文提示词
- 调用 LLM 生成基于检索结果的回答

### 4. 批量测试 (main.py)

批量处理 questions.txt 中的测试问题：

```bash
uv run main.py
```

该脚本的功能：

- 读取 `questions.txt` 中的所有测试问题
- 随机打乱问题顺序
- 逐个进行 RAG 查询并显示结果
- 支持交互式控制（回车继续，'q' 退出）

## 🏗️ 项目结构

```
rag-milvus/
├── data/                   # 文档数据目录
│   └── 001.txt             # 示例文档
├── questions.txt           # 测试问题列表
├── main.py                 # 批量测试脚本
├── milvus_rag_client.py    # Milvus 客户端封装
├── llm.py                  # LLM 调用封装
├── 1_add_data_to_coll.py   # 数据摄入脚本
├── 2_retrieval.py          # 检索测试脚本
├── 3_query.py              # RAG 查询脚本
├── pyproject.toml          # 项目配置 (uv)
├── uv.lock                 # 依赖锁定文件 (uv)
├── .env                    # 环境变量配置
└── README.md               # 项目说明
```

## 📚 核心组件

### MilvusRagClient

封装了 Milvus Lite 的所有操作：

- 集合管理 (创建、删除、查询)
- 向量存储和检索
- 支持动态字段和元数据过滤

### LLM 集成

- 支持 OpenAI 兼容 API
- 流式输出
- 自定义系统提示词

## 🔧 开发命令

```bash
# 运行任意 Python 脚本
uv run your_script.py

# 添加新的依赖包
uv add package_name

# 添加开发依赖
uv add --dev pytest black

# 更新所有依赖到最新版本
uv sync --upgrade

# 查看依赖树
uv tree
```

## 📝 注意事项

1. **首次运行前**: 确保 Ollama 服务正在运行且已下载 bge-m3 模型
2. **数据存储**: Milvus Lite 使用本地文件存储，默认路径为 `./milvus.db`
3. **向量维度**: bge-m3 模型输出 1024 维向量，请勿修改
4. **API 配置**: 根据你的 LLM 服务提供商调整 `.env` 中的配置

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
