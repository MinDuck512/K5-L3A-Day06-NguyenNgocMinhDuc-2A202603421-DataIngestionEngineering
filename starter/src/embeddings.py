"""Real, revision-pinned local embeddings with explicit token-limit checks."""
import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Protocol


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MAX_SEQUENCE_LENGTH = 512


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Use the same model, no prefix, float32, and L2 normalization for both inputs."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, local_files_only: bool = False,
                 batch_size: int = 16, cache_folder: Path | None = None) -> None:
        if model_name != DEFAULT_MODEL:
            raise ValueError("This experiment is pinned to the declared multilingual MiniLM model")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if cache_folder is None and sys.prefix == sys.base_prefix:
            raise ValueError("Run with the lab .venv Python so model files stay inside that environment")
        self.cache_folder = cache_folder or Path(sys.prefix) / "minilm"
        self.batch_size = batch_size
        self.model_name = model_name
        self.revision = MODEL_REVISION
        self.max_sequence_length = MAX_SEQUENCE_LENGTH

        # Import lazily: validation and chunking do not need to load PyTorch.
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
        import torch

        self.cpu_threads = min(4, os.cpu_count() or 1)
        torch.set_num_threads(self.cpu_threads)
        torch.manual_seed(42)
        if not local_files_only:
            print("Downloading/verifying the pinned embedding model (first run about 470 MB)", flush=True)
            snapshot_download(
                repo_id=model_name, revision=MODEL_REVISION,
                local_dir=str(self.cache_folder),
                allow_patterns=["config.json", "config_sentence_transformers.json", "modules.json",
                                "sentence_bert_config.json", "1_Pooling/config.json", "model.safetensors",
                                "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                                "sentencepiece.bpe.model", "unigram.json", "README.md"],
                max_workers=2,
            )
        for filename in ("modules.json", "model.safetensors", "tokenizer.json", "1_Pooling/config.json"):
            if not (self.cache_folder / filename).is_file():
                raise FileNotFoundError(f"Pinned model file is missing: {filename}. Run once without --offline.")
        download_metadata = self.cache_folder / ".cache/huggingface/download/model.safetensors.metadata"
        if not download_metadata.is_file() or download_metadata.read_text(encoding="utf-8").splitlines()[0] != MODEL_REVISION:
            raise ValueError("Cached weights do not have the declared pinned revision; run once without --offline")
        self._model = SentenceTransformer(str(self.cache_folder), device="cpu", local_files_only=True,
                                          trust_remote_code=False, prompts={"query": "", "document": ""})
        self._model.eval()
        self.default_sequence_length = int(self._model.max_seq_length)
        native_limit = int(self._model[0].auto_model.config.max_position_embeddings)
        self.native_sequence_length = native_limit
        if not 0 < MAX_SEQUENCE_LENGTH <= native_limit:
            raise ValueError("Requested sequence length exceeds the model architecture")
        self._model.max_seq_length = MAX_SEQUENCE_LENGTH
        self.dimensions = int(self._model.get_embedding_dimension())
        if self.dimensions != 384:
            raise ValueError(f"Unexpected real model dimensions: {self.dimensions}")
        print(f"Model ready: {self.dimensions} dimensions; limit {self.max_sequence_length} tokens; CPU", flush=True)

    def token_counts(self, texts: list[str]) -> list[int]:
        """Count the real tokenizer output including special tokens; never truncate."""
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be non-empty strings")
        if not texts:
            return []
        tokens = self._model.tokenizer(texts, add_special_tokens=True, truncation=False,
                                       padding=False, verbose=False)
        return [len(ids) for ids in tokens["input_ids"]]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        counts = self.token_counts(texts)
        over_limit = [(index, count) for index, count in enumerate(counts)
                      if count > self.max_sequence_length]
        if over_limit:
            raise ValueError(f"Refusing silent truncation: token limit {self.max_sequence_length}; inputs {over_limit[:5]}")
        if not texts:
            return []
        vectors = self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=True,
                                      precision="float32", convert_to_numpy=True,
                                      normalize_embeddings=True, prompt="")
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(texts), self.dimensions) or not np.isfinite(vectors).all():
            raise ValueError("Model returned invalid vectors or dimensions")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
            raise ValueError("Model vectors are not L2-normalized")
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, question: str) -> list[float]:
        return self._encode([question])[0]

    def describe(self) -> dict:
        with (self.cache_folder / "model.safetensors").open("rb") as weights:
            weights_sha256 = hashlib.file_digest(weights, "sha256").hexdigest()
        return {
            "provider": "Sentence Transformers local", "model_name": self.model_name,
            "revision": self.revision, "weights_sha256": weights_sha256,
            "dimensions": self.dimensions, "device": "cpu", "cpu_threads": self.cpu_threads,
            "dtype": "float32", "normalize_embeddings": True, "seed": 42,
            "default_sequence_length": self.default_sequence_length,
            "native_sequence_length": self.native_sequence_length,
            "configured_sequence_length": self.max_sequence_length,
            "sequence_length_note": "Explicit override of the 128-token model default to its native 512-token capacity; same for all strategies; quality must be judged on dev.",
            "document_processing": "Original chunk text, no prefix, mean pooling from the saved model, L2 normalization",
            "query_processing": "Question text only, no gold/filter/prefix, same model and normalization",
            "cache_folder": str(self.cache_folder), "paid_api_cost_usd": 0.0,
            "model_execution": "local; source documents are not sent to an embedding API",
            "versions": {package: version(package) for package in
                         ("sentence-transformers", "transformers", "torch", "numpy", "huggingface-hub")},
        }


def validate_embedding_artifact(data_dir: Path, results_dir: Path) -> dict:
    """Check saved vectors, input hashes, IDs, token limits and dev-only queries."""
    import numpy as np

    from .evaluation import validate_benchmark

    validate_benchmark(data_dir)
    chunk_path = results_dir / "chunking.json"
    chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
    benchmark_path = data_dir / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    dev_ids = [question["id"] for question in benchmark["questions"] if question["split"] == "dev"]
    with np.load(results_dir / "embeddings.npz", allow_pickle=False) as saved:
        metadata = json.loads(saved["metadata"].item())
        for field, path in (("chunking_sha256", chunk_path), ("benchmark_sha256", benchmark_path),
                            ("corpus_units_sha256", data_dir / "source_units.jsonl")):
            if metadata[field] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"Embedding inputs changed: {path.name}")
        model = metadata["model"]
        if (model["model_name"], model["revision"], model["dimensions"]) != (DEFAULT_MODEL, MODEL_REVISION, 384):
            raise ValueError("Saved model does not match the experiment")
        for key in ("fixed", "paragraph", "dev_query"):
            ids = saved[key + "_ids"].tolist()
            vectors = saved[key + "_vectors"]
            counts = saved[key + "_token_counts"]
            expected_ids = dev_ids if key == "dev_query" else [chunk["chunk_id"] for chunk in chunks["chunk_sets"][key]]
            if ids != expected_ids or len(set(ids)) != len(ids):
                raise ValueError(f"Vector/ID alignment failed: {key}")
            if vectors.shape != (len(ids), model["dimensions"]) or vectors.dtype != np.float32:
                raise ValueError(f"Incorrect vector shape/dtype: {key}")
            if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
                raise ValueError(f"Invalid or non-normalized vectors: {key}")
            if counts.shape != (len(ids),) or not ((counts > 0) & (counts <= model["configured_sequence_length"])).all():
                raise ValueError(f"Invalid token counts: {key}")
        if metadata["strategy_chunk_sets"] != {name: config["chunk_set"] for name, config in chunks["strategies"].items()}:
            raise ValueError("Saved strategy mappings differ from chunk configurations")
        return {
            "document_vectors": sum(len(saved[key + "_ids"]) for key in ("fixed", "paragraph")),
            "vectors_by_strategy": {name: len(saved[config["chunk_set"] + "_ids"])
                                    for name, config in chunks["strategies"].items()},
            "dev_query_vectors": len(dev_ids), "test_query_vectors": 0,
            "dimensions": model["dimensions"], "configured_token_limit": model["configured_sequence_length"],
            "max_document_tokens": max(int(saved[key + "_token_counts"].max()) for key in ("fixed", "paragraph")),
            "max_dev_query_tokens": int(saved["dev_query_token_counts"].max()),
            "vectors_finite_and_normalized": True, "inputs_match": True,
            "paid_api_cost_usd": model["paid_api_cost_usd"], "retrieval_run": False,
        }
