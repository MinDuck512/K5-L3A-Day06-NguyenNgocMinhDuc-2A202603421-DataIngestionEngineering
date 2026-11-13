"""Persistent Chroma storage; use saved embeddings and native cosine search/filter."""
from pathlib import Path
from typing import Any

from .models import Chunk


HNSW_CONFIG = {
    "space": "cosine", "ef_construction": 100, "ef_search": 100,
    "max_neighbors": 16, "resize_factor": 1.2, "sync_threshold": 100,
}
COLLECTIONS = {"A": "day6_fixed", "B": "day6_paragraph", "C": "day6_paragraph"}


class ChromaVectorStore:
    """Own two lab collections; B/C share exactly the same stored data."""

    def __init__(self, persist_dir: Path, dimensions: int, embedding_fingerprint: str,
                 *, create: bool = True) -> None:
        if type(dimensions) is not int or dimensions <= 0:
            raise ValueError("Vector dimensions must be a positive integer")
        if not create and not (persist_dir / "chroma.sqlite3").is_file():
            raise FileNotFoundError("Persistent Chroma database is missing; run --ingest-store first")
        import chromadb
        from chromadb.config import Settings

        self.persist_dir = persist_dir
        self.dimensions = dimensions
        self.embedding_fingerprint = embedding_fingerprint
        self.client = chromadb.PersistentClient(path=str(persist_dir),
                                               settings=Settings(anonymized_telemetry=False))
        self._collections = {}
        try:
            self.create_schema(create=create)
        except Exception:
            self.close()
            raise

    def create_schema(self, *, create: bool = True) -> None:
        expected = {"dimensions": self.dimensions, "embedding_fingerprint": self.embedding_fingerprint,
                    "metric": "cosine", "schema_version": 1}
        for name in sorted(set(COLLECTIONS.values())):
            if create:
                collection = self.client.get_or_create_collection(
                    name=name, embedding_function=None, metadata=expected,
                    configuration={"hnsw": dict(HNSW_CONFIG)},
                )
            else:
                collection = self.client.get_collection(name, embedding_function=None)
            if any((collection.metadata or {}).get(key) != value for key, value in expected.items()):
                raise ValueError(f"Collection {name} uses different model settings/dimensions")
            actual_config = collection.configuration.get("hnsw") or {}
            if any(actual_config.get(key) != value for key, value in HNSW_CONFIG.items()):
                raise ValueError(f"Collection {name} has a different cosine/HNSW configuration")
            self._collections[name] = collection

    def _collection(self, strategy: str):
        if strategy not in COLLECTIONS:
            raise ValueError(f"Unknown strategy: {strategy}")
        return self._collections[COLLECTIONS[strategy]]

    def _vectors(self, vectors, expected_count: int):
        import numpy as np

        values = np.asarray(vectors, dtype=np.float32)
        if values.shape != (expected_count, self.dimensions) or not np.isfinite(values).all():
            raise ValueError("Vector count/dimensions/values are invalid")
        if (np.linalg.norm(values, axis=1) == 0).any():
            raise ValueError("Cosine search does not accept zero vectors")
        return values

    @staticmethod
    def metadata_for(chunk: Chunk) -> dict[str, Any]:
        metadata = {**chunk.metadata, "document_id": chunk.document_id,
                    "source": chunk.source, "page": chunk.page}
        if not chunk.document_id or not chunk.source or type(chunk.page) is not int:
            raise ValueError("Chunk has invalid required source metadata")
        if type(metadata.get("chapter")) is not int:
            raise ValueError("Chunk requires a verified integer chapter filter field")
        if any(not isinstance(value, (str, int, float, bool)) for value in metadata.values()):
            raise ValueError("This lab stores only scalar metadata values")
        return metadata

    def count(self, strategy: str) -> int:
        return self._collection(strategy).count()

    def records(self, strategy: str, *, include_vectors: bool = False) -> dict:
        include = ["documents", "metadatas"]
        if include_vectors:
            include.append("embeddings")
        return self._collection(strategy).get(include=include)

    def upsert(self, strategy: str, chunks: list[Chunk], vectors) -> dict:
        """Synchronize a COMPLETE chunk-set snapshot, not a partial document batch.

        Upsert current stable IDs first. Then delete obsolete IDs from this lab
        collection, including chunks of removed/changed documents. Empty
        snapshots are refused to prevent accidental clearing after a load error.
        """
        if not chunks:
            raise ValueError("Refusing to synchronize an empty snapshot")
        values = self._vectors(vectors, len(chunks))
        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids) or any(not chunk.text.strip() for chunk in chunks):
            raise ValueError("Duplicate IDs or empty chunks")
        metadatas = [self.metadata_for(chunk) for chunk in chunks]
        collection = self._collection(strategy)
        old_ids = set(collection.get(include=[])["ids"])
        batch_size = min(128, self.client.get_max_batch_size())
        for start in range(0, len(chunks), batch_size):
            stop = start + batch_size
            collection.upsert(ids=ids[start:stop], embeddings=values[start:stop].tolist(),
                              documents=[chunk.text for chunk in chunks[start:stop]],
                              metadatas=metadatas[start:stop])
        obsolete = sorted(old_ids - set(ids))
        for start in range(0, len(obsolete), batch_size):
            collection.delete(ids=obsolete[start:start + batch_size])
        actual_ids = set(collection.get(include=[])["ids"])
        if actual_ids != set(ids):
            raise RuntimeError("Stored IDs differ from the complete input snapshot")
        return {"before": len(old_ids), "after": len(actual_ids),
                "new_ids": len(set(ids) - old_ids), "obsolete_removed": len(obsolete)}

    def search(self, strategy: str, query_vector, top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Use Chroma query_embeddings and where; return raw distance and 1-distance."""
        import math

        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if filters is not None:
            if (strategy != "C" or not isinstance(filters, dict) or set(filters) != {"chapter"}
                    or type(filters["chapter"]) is not int):
                raise ValueError("Only C accepts a question-provided integer chapter filter")
        vector = self._vectors([query_vector], 1)[0]
        collection = self._collection(strategy)
        if collection.count() == 0:
            return []
        result = collection.query(query_embeddings=[vector.tolist()],
                                  n_results=min(top_k, collection.count()), where=filters,
                                  include=["documents", "metadatas", "distances"])
        hits = []
        for rank, chunk_id in enumerate(result["ids"][0], start=1):
            index = rank - 1
            distance = float(result["distances"][0][index])
            if not math.isfinite(distance):
                raise RuntimeError("Chroma returned a non-finite cosine distance")
            metadata = result["metadatas"][0][index]
            if filters and metadata["chapter"] != filters["chapter"]:
                raise RuntimeError("Backend result violates the requested chapter filter")
            hits.append({"rank": rank, "chunk_id": chunk_id,
                         "document_id": metadata["document_id"], "source": metadata["source"],
                         "page": metadata["page"], "text": result["documents"][0][index],
                         "metadata": metadata, "cosine_distance": distance, "score": 1.0 - distance})
        return hits

    def close(self) -> None:
        self.client.close()
