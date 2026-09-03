"""RAG 记忆层:向量库(手写)+ 阿里云百炼 embedding。

设计要点:
- 为什么手写向量库:数据量小(几百条历史)时,暴力扫描 + 余弦相似度
  完全够用,不需要引入 FAISS 等重型框架。手写意味着底层原理必须
  自己实现一遍——理解到位,才知道何时才需要上 FAISS 这类 ANN 索引。
- embedding 走阿里云百炼的 OpenAI 兼容端点(https://dashscope.aliyuncs.com
  /compatible-mode/v1),复用 openai SDK,不引额外依赖。
- RAG 解决的问题:摘要记忆"丢了细节";向量检索按需取回精确细节。
  两者并存:摘要兜底全局,向量兜底细节。
"""
import json
import os
import uuid
from pathlib import Path

import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v4")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

_embed_client = None


def _client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "未配置 DASHSCOPE_API_KEY。请在 .env 里填入阿里云百炼的 API Key:"
                " https://bailian.console.aliyun.com/?apiKey=1#/api-key"
            )
        _embed_client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)
    return _embed_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量把文本转成向量。单条时也走批量,减少 API 往返。"""
    if not texts:
        return []
    resp = _client().embeddings.create(model=EMBED_MODEL, input=texts)
    # resp.data 的顺序与输入一致,但保险起见按 index 排序。
    ordered = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in ordered]


class VectorStore:
    """最小向量库:列表存 {id, text, vector},暴力扫描余弦相似度。

    设计要点:这本质就是 FAISS/Qdrant 在数据量小时做的事;
    FAISS 是在此基础上做 ANN(近似最近邻)索引加速大规模检索。
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.items: list[dict] = []  # [{"id", "text", "vector"}]
        self._load()

    def _load(self):
        if self.path.exists():
            self.items = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.items, ensure_ascii=False),
            encoding="utf-8",
        )

    def add(self, text: str, vector: list[float]):
        self.items.append({"id": uuid.uuid4().hex, "text": text, "vector": vector})

    def _scores(self, query: list[float]) -> list[tuple[float, int]]:
        q = np.asarray(query, dtype=float)
        out = []
        for i, item in enumerate(self.items):
            v = np.asarray(item["vector"], dtype=float)
            denom = np.linalg.norm(q) * np.linalg.norm(v)
            score = float(np.dot(q, v) / denom) if denom else 0.0
            out.append((score, i))
        return out

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """给定查询文本,返回最相关的 top_k 条记忆。"""
        if not self.items:
            return []
        [query_vec] = embed_texts([query])
        scores = sorted(self._scores(query_vec), key=lambda t: t[0], reverse=True)
        return [
            {"text": self.items[i]["text"], "score": round(s, 4)}
            for s, i in scores[:top_k]
        ]

    def __len__(self):
        return len(self.items)
