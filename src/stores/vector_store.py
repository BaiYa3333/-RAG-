"""ChromaDB 向量存储 — AsyncHttpClient 封装."""

from chromadb import AsyncHttpClient

from src.config import settings
from src.utils.logger import logger


class VectorStore:
    def __init__(self):
        self._client: AsyncHttpClient | None = None

    async def connect(self) -> None:
        self._client = await AsyncHttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        logger.info("chromadb_connected", host=settings.chroma_host, port=settings.chroma_port)

    async def close(self) -> None:
        if self._client:
            self._client = None
        logger.info("chromadb_closed")

    async def heartbeat(self) -> int:
        return await self._client.heartbeat()

    async def get_or_create_collection(self, name: str):
        return await self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(self, collection, ids: list[str], embeddings: list[list[float]],
                  metadatas: list[dict] | None = None,
                  documents: list[str] | None = None) -> None:
        await collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    async def search(self, collection, query_embeddings: list[list[float]],
                     top_k: int = 10, where: dict | None = None) -> dict:
        return await collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    async def get_by_metadata(self, metadata_filter: dict, top_k: int = 10,
                             collection_name: str = "rag_docs_dev") -> list[dict]:
        """通过元数据过滤获取文档 chunk（不需要 embedding 查询）。

        Args:
            metadata_filter: ChromaDB where 条件，如 {"source": "doc.pdf"}
            top_k: 返回数量上限
            collection_name: collection 名称，默认使用当前索引集合 rag_docs_dev

        Returns:
            [{"content": "...", "metadata": {...}, "id": "..."}, ...]
        """
        collection = await self.get_or_create_collection(collection_name)
        results = await collection.get(
            where=metadata_filter,
            limit=top_k,
            include=["documents", "metadatas"],
        )
        items: list[dict] = []
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        for i, doc_id in enumerate(ids):
            items.append({
                "id": doc_id,
                "content": documents[i] if documents and i < len(documents) else "",
                "metadata": metadatas[i] if metadatas and i < len(metadatas) else {},
            })
        return items

    async def delete(self, collection, ids: list[str]) -> None:
        await collection.delete(ids=ids)

    async def delete_where(self, collection, where: dict) -> None:
        await collection.delete(where=where)

    async def delete_collection(self, name: str) -> None:
        if self._client is None:
            raise RuntimeError("ChromaDB is not connected")
        await self._client.delete_collection(name)
