"""Run validation, chunking, local embedding, persistent storage and dev retrieval."""
import argparse
import hashlib
import json
import os
import platform
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from importlib.metadata import version


# Starting parameters, selected before retrieval; B and C differ only in filtering.
STRATEGIES = {
    "A": {"chunk_set": "fixed", "chunking": "fixed_size", "chunk_size": 500,
          "overlap": 50, "metadata_filter": False},
    "B": {"chunk_set": "paragraph", "chunking": "paragraph", "chunk_size": 500,
          "overlap": 0, "metadata_filter": False},
    "C": {"chunk_set": "paragraph", "chunking": "paragraph", "chunk_size": 500,
          "overlap": 0, "metadata_filter": True},
}


def build_chunk_artifact(data_dir: Path, results_dir: Path) -> dict:
    """Create one necessary artifact containing configurations, chunks, and stats."""
    from .chunking import by_paragraph, chunk_statistics, fixed_size
    from .evaluation import validate_benchmark
    from .ingestion import load_source_units

    validate_benchmark(data_dir)
    pages = load_source_units(data_dir / "source_units.jsonl")
    b_config = {key: value for key, value in STRATEGIES["B"].items() if key != "metadata_filter"}
    c_config = {key: value for key, value in STRATEGIES["C"].items() if key != "metadata_filter"}
    if b_config != c_config or STRATEGIES["B"]["metadata_filter"] or not STRATEGIES["C"]["metadata_filter"]:
        raise ValueError("B and C must differ only in metadata filtering")

    chunk_sets = {}
    summaries = {}
    for name in ("A", "B"):
        config = STRATEGIES[name]

        def build(config=config):
            if config["chunking"] == "fixed_size":
                return [chunk for page in pages for chunk in
                        fixed_size(page, config["chunk_size"], config["overlap"])]
            return [chunk for page in pages for chunk in by_paragraph(page, config["chunk_size"])]

        started = perf_counter()
        chunks = build()
        seconds = perf_counter() - started
        summary = chunk_statistics(pages, chunks, config["chunk_size"])
        rebuilt = build()
        if [asdict(chunk) for chunk in chunks] != [asdict(chunk) for chunk in rebuilt]:
            raise ValueError("Rebuilding identical input changed chunk IDs, text, or metadata")
        summary.update({"chunking_seconds": round(seconds, 6), "deterministic_rebuild_verified": True})
        chunk_sets[config["chunk_set"]] = chunks
        summaries[name] = summary
    summaries["C"] = {**summaries["B"], "chunking_seconds": None, "reuses_chunks_from": "B"}

    # Reserve the same ten source positions before any manual coherence scores exist.
    ordered_pages = sorted(pages, key=lambda page: (page.document_id, page.page))
    sample_count = min(10, len(ordered_pages))
    selected_indices = ([index * (len(ordered_pages) - 1) // (sample_count - 1)
                         for index in range(sample_count)] if sample_count > 1 else [0])
    samples = []
    for sample_number, page_index in enumerate(selected_indices, start=1):
        page = ordered_pages[page_index]
        middle = len(page.text) // 2
        position = min((index for index, character in enumerate(page.text) if not character.isspace()),
                       key=lambda index: (abs(index - middle), index))
        chunk_ids = {}
        for name, config in STRATEGIES.items():
            covering = [chunk for chunk in chunk_sets[config["chunk_set"]]
                        if (chunk.document_id, chunk.page) == (page.document_id, page.page)
                        and chunk.metadata["unit_char_start"] <= position < chunk.metadata["unit_char_end"]]
            if not covering:
                raise ValueError("No chunk covers a selected coherence position")
            chunk_ids[name] = min(covering, key=lambda chunk: chunk.metadata["unit_char_start"]).chunk_id
        samples.append({
            "sample_id": f"coherence-{sample_number:02d}", "document_id": page.document_id,
            "source": page.source, "page": page.page, "unit_id": page.metadata["unit_id"],
            "unit_char_position": position,
            "document_char_position": page.metadata["source_char_start"] + position,
            "chunk_id_by_strategy": chunk_ids,
        })

    artifact = {
        "stage": "chunking_only",
        "created_at": datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "corpus_units_sha256": hashlib.sha256((data_dir / "source_units.jsonl").read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256((data_dir / "benchmark.json").read_bytes()).hexdigest(),
        "input_source_units": len(pages),
        "strategies": STRATEGIES,
        "planned_retrieval": {
            "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "vector_store": "Chroma", "metric": "cosine", "top_k": 3,
            "status": "not_run", "token_count": None,
            "token_limit_check": "Required when the embedding model is loaded; no truncation check has run yet",
        },
        "comparison_notes": [
            "A-B compares chunking policy including the overlap change; it does not isolate overlap from paragraph boundaries.",
            "B-C uses identical chunk IDs, text, and metadata; only question-provided chapter filtering will differ at retrieval time.",
            "Character statistics and chunking time are measured here; embedding time, full ingest time, costs, and retrieval metrics are not measured yet.",
        ],
        "summary": summaries,
        "coherence_sampling": {
            "rule": "Sort source units by document_id/page; take floor(i*(N-1)/9) for i=0..9; choose nearest non-whitespace character to each midpoint. For overlapping chunks choose the earliest start.",
            "status": "positions_selected_labels_not_assigned", "positions": samples,
        },
        "chunk_sets": {name: [asdict(chunk) for chunk in chunks] for name, chunks in chunk_sets.items()},
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    destination = results_dir / "chunking.json"
    destination.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "source_units": len(pages),
        "chunks_by_strategy": {name: summary["chunk_count"] for name, summary in summaries.items()},
        "stored_chunk_sets": len(chunk_sets), "B_C_identical_chunks": True,
        "coherence_positions": len(samples), "output": str(destination),
    }


def build_embedding_artifact(data_dir: Path, results_dir: Path, *, offline: bool = False) -> dict:
    """Encode both chunk sets and dev questions; never retrieve or encode test queries."""
    import numpy as np

    from .chunking import chunk_statistics
    from .embeddings import MyEmbedder, validate_embedding_artifact
    from .evaluation import validate_benchmark
    from .ingestion import load_source_units
    from .models import Chunk

    validate_benchmark(data_dir)
    chunk_path = results_dir / "chunking.json"
    chunk_bytes = chunk_path.read_bytes()
    artifact = json.loads(chunk_bytes)
    benchmark_path = data_dir / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    units_path = data_dir / "source_units.jsonl"
    if (artifact["corpus_units_sha256"] != hashlib.sha256(units_path.read_bytes()).hexdigest()
            or artifact["benchmark_sha256"] != hashlib.sha256(benchmark_path.read_bytes()).hexdigest()):
        raise ValueError("Chunk artifact does not match the frozen corpus/benchmark")
    if artifact["strategies"] != STRATEGIES:
        raise ValueError("Chunking parameters changed; explicitly rebuild chunks before embedding")
    pages = load_source_units(units_path)
    for key, chunks in artifact["chunk_sets"].items():
        chunk_statistics(pages, [Chunk(**chunk) for chunk in chunks],
                         next(config["chunk_size"] for config in STRATEGIES.values() if config["chunk_set"] == key))

    started = perf_counter()
    embedder = MyEmbedder(local_files_only=offline)
    model_prepare_seconds = perf_counter() - started
    arrays = {}
    token_statistics = {}
    encoding_seconds = {}
    for key, chunks in artifact["chunk_sets"].items():
        texts = [chunk["text"] for chunk in chunks]
        counts = embedder.token_counts(texts)
        token_statistics[key] = {
            "texts": len(texts), "total_tokens_including_special": sum(counts),
            "min_tokens": min(counts), "max_tokens": max(counts),
            "over_default_128_tokens": sum(count > embedder.default_sequence_length for count in counts),
            "over_configured_limit": sum(count > embedder.max_sequence_length for count in counts),
        }
        print(f"Encoding {key}: {len(texts)} chunks; longest input {max(counts)} tokens", flush=True)
        started = perf_counter()
        vectors = np.asarray(embedder.embed_documents(texts), dtype=np.float32)
        encoding_seconds[key] = round(perf_counter() - started, 6)
        arrays[key + "_ids"] = np.asarray([chunk["chunk_id"] for chunk in chunks], dtype=np.str_)
        arrays[key + "_vectors"] = vectors
        arrays[key + "_token_counts"] = np.asarray(counts, dtype=np.int32)

    dev_questions = [question for question in benchmark["questions"] if question["split"] == "dev"]
    question_texts = [question["question"] for question in dev_questions]
    counts = embedder.token_counts(question_texts)
    print(f"Encoding {len(dev_questions)} dev queries; test queries are held out", flush=True)
    started = perf_counter()
    arrays["dev_query_vectors"] = np.asarray(embedder.embed_documents(question_texts), dtype=np.float32)
    encoding_seconds["dev_queries"] = round(perf_counter() - started, 6)
    arrays["dev_query_ids"] = np.asarray([question["id"] for question in dev_questions], dtype=np.str_)
    arrays["dev_query_token_counts"] = np.asarray(counts, dtype=np.int32)
    token_statistics["dev_queries"] = {
        "texts": len(question_texts), "total_tokens_including_special": sum(counts),
        "min_tokens": min(counts), "max_tokens": max(counts),
        "over_configured_limit": sum(count > embedder.max_sequence_length for count in counts),
    }
    # Check that the single-query method has exactly the same preprocessing as batches.
    first_query = np.asarray(embedder.embed_query(question_texts[0]), dtype=np.float32)
    if not np.allclose(first_query, arrays["dev_query_vectors"][0], atol=1e-5):
        raise ValueError("Single-query and batched query embeddings differ")
    # Validate overflow rejection with a synthetic input kept only in RAM.
    try:
        embedder.embed_query("hello " * (embedder.max_sequence_length + 1))
    except ValueError as error:
        if "Refusing silent truncation" not in str(error):
            raise
    else:
        raise ValueError("The tokenizer overflow guard did not reject oversized input")

    metadata = {
        "stage": "embeddings_only",
        "created_at": datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds"),
        "python_version": platform.python_version(), "model": embedder.describe(),
        "chunking_sha256": hashlib.sha256(chunk_bytes).hexdigest(),
        "corpus_units_sha256": hashlib.sha256(units_path.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
        "strategy_chunk_sets": {name: config["chunk_set"] for name, config in STRATEGIES.items()},
        "strategies": STRATEGIES, "token_statistics": token_statistics,
        "timing_seconds": {"model_prepare_download_and_load": round(model_prepare_seconds, 6),
                           "encoding": encoding_seconds},
        "checks": {"single_and_batch_query_match": True, "oversized_input_rejected": True},
        "test_queries_encoded": False, "retrieval_run": False,
        "limitations": [
            "Sequence length is explicitly set to 512 instead of the model's default 128; evaluate quality on dev.",
            "This NumPy file is an intermediate artifact, not the required persistent vector store.",
            "No retrieval accuracy or human coherence/grounding labels are measured at this stage.",
        ],
    }
    arrays["metadata"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    destination = results_dir / "embeddings.npz"
    np.savez_compressed(destination, **arrays)
    summary = validate_embedding_artifact(data_dir, results_dir)
    summary["output"] = str(destination)
    return summary


def _store_inputs(data_dir: Path, results_dir: Path):
    """Load verified saved inputs; this stage never loads an embedding model."""
    import numpy as np

    from .embeddings import validate_embedding_artifact

    validate_embedding_artifact(data_dir, results_dir)
    chunks = json.loads((results_dir / "chunking.json").read_text(encoding="utf-8"))
    if chunks["strategies"] != STRATEGIES:
        raise ValueError("Stored chunks do not match current strategy settings")
    benchmark = json.loads((data_dir / "benchmark.json").read_text(encoding="utf-8"))
    with np.load(results_dir / "embeddings.npz", allow_pickle=False) as saved:
        metadata = json.loads(saved["metadata"].item())
        arrays = {key: saved[key].copy() for key in saved.files if key != "metadata"}
    fingerprint_keys = ("model_name", "revision", "weights_sha256", "dimensions", "dtype",
                        "normalize_embeddings", "configured_sequence_length", "document_processing",
                        "query_processing", "versions")
    fingerprint = hashlib.sha256(json.dumps({key: metadata["model"][key] for key in fingerprint_keys},
                                            sort_keys=True).encode("utf-8")).hexdigest()
    return chunks, benchmark, metadata, arrays, fingerprint


def _verify_store_records(store, artifact: dict, arrays: dict) -> int:
    """Compare every stored text, scalar metadata field and vector to saved inputs."""
    import numpy as np

    from .models import Chunk

    checked = 0
    for strategy in ("A", "B"):
        key = STRATEGIES[strategy]["chunk_set"]
        records = store.records(strategy, include_vectors=True)
        index_by_id = {chunk_id: index for index, chunk_id in enumerate(records["ids"])}
        expected = artifact["chunk_sets"][key]
        if set(index_by_id) != {chunk["chunk_id"] for chunk in expected}:
            raise ValueError("Persistent collection IDs do not match the input snapshot")
        for index, value in enumerate(expected):
            chunk = Chunk(**value)
            stored_index = index_by_id[chunk.chunk_id]
            if (records["documents"][stored_index] != chunk.text
                    or records["metadatas"][stored_index] != store.metadata_for(chunk)):
                raise ValueError("Stored chunk content/metadata differs from source")
            if not np.allclose(records["embeddings"][stored_index], arrays[key + "_vectors"][index], atol=1e-5):
                raise ValueError("Stored embedding differs from its saved source vector")
            checked += 1
    if store.count("C") != store.count("B"):
        raise ValueError("B/C do not share the same stored data")
    return checked


def _dev_store_queries(store, artifact: dict, benchmark: dict, arrays: dict):
    """Run the same cached dev vectors against A/B/C; validate mechanics, not gold."""
    expected_chunks = {chunk["chunk_id"]: chunk for chunks in artifact["chunk_sets"].values() for chunk in chunks}
    vectors = dict(zip(arrays["dev_query_ids"].tolist(), arrays["dev_query_vectors"]))
    traces = []
    filtered_questions = 0
    identical_unfiltered = 0
    verified_records = 0
    for question in benchmark["questions"]:
        if question["split"] != "dev":
            continue
        hits_by_strategy = {}
        for strategy, config in STRATEGIES.items():
            filters = question.get("filter") if config["metadata_filter"] else None
            started = perf_counter()
            hits = store.search(strategy, vectors[question["id"]], top_k=3, filters=filters)
            elapsed = perf_counter() - started
            if len(hits) != 3:
                raise RuntimeError(f"Expected 3 dev results for {question['id']}/{strategy}, got {len(hits)}")
            for hit in hits:
                expected = expected_chunks.get(hit["chunk_id"])
                if expected is None or any(hit[field] != expected[field] for field in
                                           ("text", "document_id", "source", "page")):
                    raise RuntimeError("Retrieved record cannot be traced to the saved source chunk")
                if any(hit["metadata"].get(key) != value for key, value in expected["metadata"].items()):
                    raise RuntimeError("Retrieved metadata differs from source")
                verified_records += 1
            hits_by_strategy[strategy] = [hit["chunk_id"] for hit in hits]
            traces.append({
                "question_id": question["id"], "split": "dev", "strategy": strategy,
                "question": question["question"], "requested_filter": question.get("filter"),
                "applied_filter": filters, "query_input": "cached embedding of question text only",
                "query_seconds": round(elapsed, 6), "hits": hits,
            })
            if strategy == "C" and filters:
                if not all(hit["metadata"]["chapter"] == filters["chapter"] for hit in hits):
                    raise RuntimeError("Filtered C results are outside the requested chapter")
                filtered_questions += 1
        if question.get("filter") is None:
            if hits_by_strategy["B"] != hits_by_strategy["C"]:
                raise RuntimeError("Unfiltered B/C differ despite sharing data and query vector")
            identical_unfiltered += 1
    return traces, {
        "dev_query_strategy_runs": len(traces), "retrieved_records_verified": verified_records,
        "filtered_C_questions": filtered_questions,
        "filtered_C_questions_obeying_filter": filtered_questions,
        "unfiltered_B_C_queries_with_identical_top3": identical_unfiltered,
        "test_queries_run": 0, "gold_accuracy_measured": False,
    }


def ingest_store(data_dir: Path, results_dir: Path) -> dict:
    """Persist vectors, exercise idempotence/obsolete cleanup, and save dev top-3."""
    from .models import Chunk
    from .store import ChromaVectorStore, COLLECTIONS, HNSW_CONFIG

    artifact, benchmark, embedding, arrays, fingerprint = _store_inputs(data_dir, results_dir)
    previous_report = results_dir / "retrieval.json"
    previous = json.loads(previous_report.read_text(encoding="utf-8")) if previous_report.is_file() else {}
    setup_notes = previous.get("setup_notes", [])
    frozen = previous.get("selection", {}).get("frozen", False)
    if frozen:
        current_hashes = {"chunking_sha256": hashlib.sha256((results_dir / "chunking.json").read_bytes()).hexdigest(),
                          "embeddings_sha256": hashlib.sha256((results_dir / "embeddings.npz").read_bytes()).hexdigest(),
                          "benchmark_sha256": hashlib.sha256((data_dir / "benchmark.json").read_bytes()).hexdigest()}
        if (previous["input_hashes"] != current_hashes or previous["strategies"] != STRATEGIES
                or previous["store"]["embedding_fingerprint"] != fingerprint or previous["store"]["hnsw"] != HNSW_CONFIG):
            raise ValueError("Inputs/settings changed after test freeze; preserve this experiment and start a separately documented experiment")
    opened = perf_counter()
    store = ChromaVectorStore(results_dir / "chroma", embedding["model"]["dimensions"], fingerprint)
    open_seconds = perf_counter() - opened
    first_upsert, second_upsert, upsert_seconds = {}, {}, {}
    chunk_sets = {name: [Chunk(**value) for value in artifact["chunk_sets"][STRATEGIES[name]["chunk_set"]]]
                  for name in ("A", "B")}
    try:
        for strategy, chunks in chunk_sets.items():
            started = perf_counter()
            first_upsert[strategy] = store.upsert(strategy, chunks, arrays[STRATEGIES[strategy]["chunk_set"] + "_vectors"])
            upsert_seconds[strategy] = round(perf_counter() - started, 6)
        # Real-data functional check: remove one existing chunk from a temporary
        # snapshot, verify obsolete deletion, and restore the complete snapshot.
        chunks_a = chunk_sets["A"]
        try:
            reduced = store.upsert("A", chunks_a[1:], arrays["fixed_vectors"][1:])
            if reduced["after"] != len(chunks_a) - 1 or reduced["obsolete_removed"] != 1:
                raise RuntimeError("Obsolete chunk cleanup check failed")
            if chunks_a[0].chunk_id in store.records("A")["ids"]:
                raise RuntimeError("The omitted old chunk is still stored")
        finally:
            restored = store.upsert("A", chunks_a, arrays["fixed_vectors"])
        stale_check = {
            "method": "Temporarily omit one real chunk from A's complete snapshot, then restore it; source files unchanged",
            "temporarily_removed_id": chunks_a[0].chunk_id,
            "reduced_count": reduced["after"], "obsolete_removed": reduced["obsolete_removed"],
            "restored_count": restored["after"],
        }
        for strategy, chunks in chunk_sets.items():
            second_upsert[strategy] = store.upsert(strategy, chunks, arrays[STRATEGIES[strategy]["chunk_set"] + "_vectors"])
            if second_upsert[strategy]["before"] != len(chunks) or second_upsert[strategy]["after"] != len(chunks):
                raise RuntimeError("Re-ingesting the same snapshot changed record counts")
        verified_records = _verify_store_records(store, artifact, arrays)
        traces, query_checks = _dev_store_queries(store, artifact, benchmark, arrays)
        counts = {name: store.count(name) for name in STRATEGIES}
    finally:
        store.close()
    database_bytes = sum(path.stat().st_size for path in (results_dir / "chroma").rglob("*") if path.is_file())
    report = {
        "stage": "persistent_store_and_dev_retrieval",
        "created_at": datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds"),
        "writer_process_id": os.getpid(),
        "store": {"backend": "Chroma PersistentClient", "version": version("chromadb"),
                  "path": "starter/results/chroma", "collections": COLLECTIONS,
                  "dimensions": embedding["model"]["dimensions"], "metric": "cosine",
                  "search_type": "approximate nearest neighbors, HNSW", "hnsw": HNSW_CONFIG,
                  "embedding_fingerprint": fingerprint, "embedding_function": None,
                  "physical_record_count": verified_records, "counts_by_strategy": counts,
                  "database_bytes_after_close": database_bytes},
        "model": embedding["model"], "strategies": STRATEGIES,
        "input_hashes": {
            "chunking_sha256": hashlib.sha256((results_dir / "chunking.json").read_bytes()).hexdigest(),
            "embeddings_sha256": hashlib.sha256((results_dir / "embeddings.npz").read_bytes()).hexdigest(),
            "benchmark_sha256": hashlib.sha256((data_dir / "benchmark.json").read_bytes()).hexdigest(),
        },
        "ingestion_checks": {"first_upsert": first_upsert, "second_upsert": second_upsert,
                             "obsolete_cleanup": stale_check, "records_verified": verified_records},
        "timing_seconds": {"open_and_schema": round(open_seconds, 6), "first_upsert": upsert_seconds},
        "dev_query_checks": query_checks, "dev_traces": traces,
        "persistence_check": {"status": "pending --check-store in a new Python process"},
        "test_queries_run": False, "retrieval_accuracy_measured": False,
        "manual_audit_labels_assigned": False,
        "setup_notes": setup_notes,
    }
    if frozen:
        old_ids = [[hit["chunk_id"] for hit in trace["hits"]] for trace in previous["dev_traces"]]
        new_ids = [[hit["chunk_id"] for hit in trace["hits"]] for trace in traces]
        previous["latest_store_reingestion"] = {
            "created_at": report["created_at"], "writer_process_id": os.getpid(),
            "counts_by_strategy": counts, "ingestion_checks": report["ingestion_checks"],
            "timing_seconds": report["timing_seconds"], "dev_top3_identical_to_frozen_traces": old_ids == new_ids,
            "frozen_selection_test_results_and_audits_preserved": True,
        }
        previous["writer_process_id"] = os.getpid()
        previous["persistence_check"] = report["persistence_check"]
        report = previous
    (results_dir / "retrieval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"counts_by_strategy": counts, "physical_records": verified_records,
            "reingest_without_duplicates": True, "obsolete_cleanup_verified": True,
            **query_checks, "output": str(results_dir / "retrieval.json"),
            "frozen_benchmark_preserved": frozen,
            "next_check": "Run --check-store in a new Python process"}


def check_store(data_dir: Path, results_dir: Path) -> dict:
    """Read the persisted database in a new process and update the same report."""
    from .store import ChromaVectorStore

    artifact, benchmark, embedding, arrays, fingerprint = _store_inputs(data_dir, results_dir)
    report_path = results_dir / "retrieval.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for field, path in (("chunking_sha256", results_dir / "chunking.json"),
                        ("embeddings_sha256", results_dir / "embeddings.npz"),
                        ("benchmark_sha256", data_dir / "benchmark.json")):
        if report["input_hashes"][field] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Store report is stale; run --ingest-store with the current inputs")
    if report["writer_process_id"] == os.getpid():
        raise RuntimeError("Persistence must be verified from a new Python process")
    store = ChromaVectorStore(results_dir / "chroma", embedding["model"]["dimensions"], fingerprint, create=False)
    try:
        checked = _verify_store_records(store, artifact, arrays)
        reopened_traces, checks = _dev_store_queries(store, artifact, benchmark, arrays)
        counts = {name: store.count(name) for name in STRATEGIES}
    finally:
        store.close()
    original_ids = [[hit["chunk_id"] for hit in trace["hits"]] for trace in report["dev_traces"]]
    reopened_ids = [[hit["chunk_id"] for hit in trace["hits"]] for trace in reopened_traces]
    report["persistence_check"] = {
        "status": "passed", "reader_process_id": os.getpid(), "fresh_process": True,
        "checked_at": datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds"),
        "counts_by_strategy": counts, "all_records_text_metadata_vectors_verified": checked,
        "dev_top3_ids_same_after_reopen": original_ids == reopened_ids,
        "dev_query_checks": checks,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report["persistence_check"]


def score_dev(data_dir: Path, results_dir: Path) -> dict:
    """Score saved dev traces without running retrieval or choosing from test."""
    from .evaluation import score_traces, summarize

    _store_inputs(data_dir, results_dir)
    path = results_dir / "retrieval.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    benchmark = json.loads((data_dir / "benchmark.json").read_text(encoding="utf-8"))
    units = [json.loads(line) for line in (data_dir / "source_units.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    report["dev_traces"] = score_traces(report["dev_traces"], benchmark["questions"], report.get("evidence_reviews", []), units)
    report["metrics"] = summarize(report["dev_traces"] + report.get("test_traces", []))
    report["retrieval_accuracy_measured"] = True
    report["evaluation_method"] = {
        "primary": "Single retrieved chunk with sufficient evidence at an accepted source/unit; literal gold or documented source-verified semantic review",
        "literal_baseline": "Complete gold quote after Unicode/whitespace normalization at the registered source/unit",
        "multiple_chunks": "Union coverage is diagnostic only; never replaces single-chunk Hit@3",
        "semantic_reviewer": "Codex AI assistant; human author review pending",
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {name: splits["dev"] for name, splits in report["metrics"].items()}


def _experiment_signature(report: dict) -> str:
    content = {"strategies": STRATEGIES, "hnsw": report["store"]["hnsw"],
               "embedding_fingerprint": report["store"]["embedding_fingerprint"],
               "input_hashes": report["input_hashes"]}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode("utf-8")).hexdigest()


def run_test(data_dir: Path, results_dir: Path) -> dict:
    """Freeze the dev decision on disk BEFORE embedding/retrieving any test query."""
    import numpy as np

    from .embeddings import MyEmbedder
    from .evaluation import score_traces, summarize
    from .store import ChromaVectorStore

    score_dev(data_dir, results_dir)
    artifact, benchmark, embedding, arrays, fingerprint = _store_inputs(data_dir, results_dir)
    path = results_dir / "retrieval.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("test_traces"):
        if report["selection"]["experiment_signature"] != _experiment_signature(report):
            raise ValueError("Frozen experiment changed after test; do not reuse its holdout claim")
        return {"already_evaluated": True, "selected_strategy": report["selection"]["selected_strategy"],
                "test_metrics": {name: splits["test"] for name, splits in report["metrics"].items()}}
    if not report.get("dev_evidence_review", {}).get("status", "").startswith("AI-assisted review completed"):
        raise ValueError("Complete/document dev evidence review before freezing test")
    previous_selection = report.get("selection")
    if previous_selection and previous_selection["experiment_signature"] != _experiment_signature(report):
        raise ValueError("The already-frozen experiment signature changed")
    candidates = []
    for strategy, config in STRATEGIES.items():
        metrics = report["metrics"][strategy]["dev"]
        timing_strategy = "B" if strategy == "C" else strategy
        seconds = report["timing_seconds"]["first_upsert"][timing_strategy]
        count = report["store"]["counts_by_strategy"][strategy]
        key = [-metrics["evidence_hit_at_3"]["fraction"], -metrics["hit_at_1"]["fraction"], seconds, count]
        candidates.append({"strategy": strategy, "ranking_key": key, "dev_metrics": metrics,
                           "measured_vector_upsert_seconds": seconds, "chunk_count": count})
    candidates.sort(key=lambda candidate: (*candidate["ranking_key"], candidate["strategy"]))
    best_key = candidates[0]["ranking_key"]
    report["selection"] = {
        "frozen": True, "frozen_at": datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="microseconds"),
        "selected_strategy": candidates[0]["strategy"],
        "co_best_strategies": [candidate["strategy"] for candidate in candidates if candidate["ranking_key"] == best_key],
        "criteria": benchmark["policy"]["selection_criteria_in_order"], "candidates": candidates,
        "timing_definition": "Stage-6 measured vector upsert seconds; excludes chunking/embedding/model load. C shares the B ingestion cost. No timing tie-break is needed if retrieval metrics differ.",
        "residual_tie_policy": report["dev_evidence_review"]["unresolved_tie_policy"],
        "experiment_signature": _experiment_signature(report),
        "test_results_available_when_selected": False,
        "label_status_when_selected": "Source-reviewed by Codex; human author confirmation pending",
    }
    if previous_selection:
        report["selection"] = previous_selection
    # Persist the decision before preparing test queries for embedding/retrieval.
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    test_started_at = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="microseconds")
    test_questions = [question for question in benchmark["questions"] if question["split"] == "test"]
    embedder = MyEmbedder(local_files_only=True)
    description = embedder.describe()
    if any(description[key] != embedding["model"][key] for key in
           ("model_name", "revision", "weights_sha256", "dimensions", "configured_sequence_length", "versions")):
        raise ValueError("Test query model differs from the document/dev model")
    texts = [question["question"] for question in test_questions]
    started = perf_counter()
    test_vectors = np.asarray(embedder.embed_documents(texts), dtype=np.float32)
    embedding_seconds = perf_counter() - started
    counts = embedder.token_counts(texts)
    store = ChromaVectorStore(results_dir / "chroma", description["dimensions"], fingerprint, create=False)
    by_id = {chunk["chunk_id"]: chunk for chunks in artifact["chunk_sets"].values() for chunk in chunks}
    traces = []
    try:
        for question, vector in zip(test_questions, test_vectors):
            for strategy, config in STRATEGIES.items():
                filters = question.get("filter") if config["metadata_filter"] else None
                started = perf_counter()
                hits = store.search(strategy, vector, top_k=3, filters=filters)
                seconds = perf_counter() - started
                if len(hits) != 3:
                    raise RuntimeError("Test did not return the required top-3")
                for hit in hits:
                    expected = by_id[hit["chunk_id"]]
                    if any(hit[field] != expected[field] for field in ("text", "source", "document_id", "page")):
                        raise RuntimeError("Test result does not match the stored source chunk")
                traces.append({"question_id": question["id"], "split": "test", "strategy": strategy,
                               "question": question["question"], "requested_filter": question.get("filter"),
                               "applied_filter": filters, "query_seconds": round(seconds, 6), "hits": hits})
    finally:
        store.close()
    units = [json.loads(line) for line in (data_dir / "source_units.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    report["test_traces"] = score_traces(traces, benchmark["questions"], report.get("evidence_reviews", []), units)
    report["test_query_embeddings"] = {
        "question_ids": [question["id"] for question in test_questions], "vectors": test_vectors.tolist(),
        "token_counts": counts, "dimensions": description["dimensions"],
        "embedding_seconds": round(embedding_seconds, 6), "embedding_fingerprint": fingerprint,
        "created_after_freeze": True,
    }
    report["test_started_at"] = test_started_at
    report["test_completed_at"] = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="microseconds")
    report["test_queries_run"] = True
    report["metrics"] = summarize(report["dev_traces"] + report["test_traces"])
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"selected_strategy_before_test": report["selection"]["selected_strategy"],
            "frozen_at": report["selection"]["frozen_at"], "test_runs": len(traces),
            "test_metrics": {name: splits["test"] for name, splits in report["metrics"].items()}}


def _metric_display(metric: dict) -> str:
    return f"{metric['hits']}/{metric['total']} ({100 * metric['fraction']:.1f}%)"


def _update_experiment_log(report: dict, path: Path) -> None:
    """Update the existing required CSV; preserve parameters and initial hypotheses."""
    import csv

    with path.open("r+", encoding="utf-8", newline="") as output:
        reader = csv.DictReader(output)
        fieldnames = reader.fieldnames
        rows = list(reader)
        if {row["strategy"] for row in rows} != set(STRATEGIES) or len(rows) != len(STRATEGIES):
            raise ValueError("Experiment log must contain one A/B/C row")
        for row in rows:
            strategy = row["strategy"]
            for split in ("dev", "test"):
                row[split + "_hit1"] = _metric_display(report["metrics"][strategy][split]["hit_at_1"])
                row[split + "_hit3"] = _metric_display(report["metrics"][strategy][split]["evidence_hit_at_3"])
            row["notes"] = (
                f"Benchmark completed; selected_by_dev={report['selection']['selected_strategy']}; "
                f"Chroma {report['store']['version']} cosine/HNSW, 384 dimensions, max_seq_length=512; "
                f"coherence_AI={report['audit_summary'][strategy]['coherence']['mean_out_of_2']}/2; "
                f"grounding_AI={report['audit_summary'][strategy]['grounding']['mean_out_of_2']}/2; "
                "literal-only metrics also retained; 2 source-checked AI evidence equivalences; "
                "rubric labels AI-assisted, human author review pending; paid_api_cost_usd=0; "
                "see starter/results/retrieval.json."
            )
        output.seek(0)
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        output.truncate()


def score_results(data_dir: Path, results_dir: Path) -> dict:
    """Recalculate saved evidence scores and aggregate explicit AI-assisted labels."""
    import numpy as np

    from .evaluation import score_traces, summarize, summarize_audits

    _store_inputs(data_dir, results_dir)
    path = results_dir / "retrieval.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("test_traces") or report["selection"]["experiment_signature"] != _experiment_signature(report):
        raise ValueError("Run the frozen test before aggregating its results")
    benchmark = json.loads((data_dir / "benchmark.json").read_text(encoding="utf-8"))
    units = [json.loads(line) for line in (data_dir / "source_units.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for split in ("dev", "test"):
        report[split + "_traces"] = score_traces(report[split + "_traces"], benchmark["questions"],
                                                report.get("evidence_reviews", []), units)
    report["metrics"] = summarize(report["dev_traces"] + report["test_traces"])
    report["audit_summary"] = summarize_audits(report["audits"])
    chunking = json.loads((results_dir / "chunking.json").read_text(encoding="utf-8"))
    with np.load(results_dir / "embeddings.npz", allow_pickle=False) as saved:
        embedding = json.loads(saved["metadata"].item())
    resources = {}
    dev_filters = {}
    for strategy, config in STRATEGIES.items():
        key = config["chunk_set"]
        timing_strategy = "B" if strategy == "C" else strategy
        resources[strategy] = {
            "chunk_count": chunking["summary"][strategy]["chunk_count"],
            "total_chunk_characters": chunking["summary"][strategy]["total_chunk_characters"],
            "document_tokens_including_special": embedding["token_statistics"][key]["total_tokens_including_special"],
            "chunking_seconds": chunking["summary"][timing_strategy]["chunking_seconds"],
            "document_embedding_seconds": embedding["timing_seconds"]["encoding"][key],
            "vector_upsert_seconds": report["timing_seconds"]["first_upsert"][timing_strategy],
            "shared_with": "B" if strategy == "C" else None,
            "paid_api_cost_usd": 0.0,
            "timing_note": "Measured stage components from recorded runs, not a single end-to-end ingest measurement",
        }
        filtered = [trace for trace in report["dev_traces"] if trace["strategy"] == strategy and trace.get("requested_filter")]
        compliant = sum(bool(trace["hits"]) and all(all(hit["metadata"].get(field) == value
                         for field, value in trace["requested_filter"].items()) for hit in trace["hits"]) for trace in filtered)
        dev_filters[strategy] = {"compliant_questions": compliant, "filter_questions": len(filtered)}
    report["resource_comparison"] = resources
    report["dev_filter_compliance"] = dev_filters
    report["stage"] = "benchmark_evaluated_author_review_pending"
    report["retrieval_accuracy_measured"] = True
    report["ai_assisted_audit_labels_assigned"] = True
    report["manual_audit_labels_assigned"] = False
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _update_experiment_log(report, data_dir.parent.parent / "artifacts/experiment_log.csv")
    return {
        "selected_before_test": report["selection"]["selected_strategy"],
        "summary": [{"strategy": name,
                     "dev_hit1": _metric_display(report["metrics"][name]["dev"]["hit_at_1"]),
                     "dev_hit3": _metric_display(report["metrics"][name]["dev"]["evidence_hit_at_3"]),
                     "test_hit1": _metric_display(report["metrics"][name]["test"]["hit_at_1"]),
                     "test_hit3": _metric_display(report["metrics"][name]["test"]["evidence_hit_at_3"]),
                     "coherence_AI_mean": report["audit_summary"][name]["coherence"]["mean_out_of_2"],
                     "grounding_AI_mean": report["audit_summary"][name]["grounding"]["mean_out_of_2"]}
                    for name in STRATEGIES],
        "human_author_review_pending": True,
    }


def validate_results(data_dir: Path, results_dir: Path) -> dict:
    """Validate all saved metrics, frozen ordering, and trace-linked audit labels offline."""
    import csv
    import numpy as np

    from .evaluation import build_metadata_audit, score_traces, summarize, summarize_audits

    artifact, benchmark, embedding, arrays, fingerprint = _store_inputs(data_dir, results_dir)
    report = json.loads((results_dir / "retrieval.json").read_text(encoding="utf-8"))
    for field, path in (("chunking_sha256", results_dir / "chunking.json"),
                        ("embeddings_sha256", results_dir / "embeddings.npz"),
                        ("benchmark_sha256", data_dir / "benchmark.json")):
        if report["input_hashes"][field] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Benchmark result inputs changed")
    traces = report["dev_traces"] + report["test_traces"]
    expected_pairs = {(question["id"], strategy) for question in benchmark["questions"] for strategy in STRATEGIES}
    pairs = [(trace["question_id"], trace["strategy"]) for trace in traces]
    if len(pairs) != len(expected_pairs) or set(pairs) != expected_pairs:
        raise ValueError("Missing or duplicate question/strategy trace")
    chunks = {chunk["chunk_id"]: chunk for values in artifact["chunk_sets"].values() for chunk in values}
    questions_by_id = {question["id"]: question for question in benchmark["questions"]}
    for trace in traces:
        question = questions_by_id[trace["question_id"]]
        expected_filter = question.get("filter") if STRATEGIES[trace["strategy"]]["metadata_filter"] else None
        if (trace["question"] != question["question"] or trace.get("requested_filter") != question.get("filter")
                or trace.get("applied_filter") != expected_filter):
            raise ValueError("Trace question/filter differs from the frozen query input")
        if [hit["rank"] for hit in trace["hits"]] != [1, 2, 3]:
            raise ValueError("Incomplete top-3 trace")
        for hit in trace["hits"]:
            chunk = chunks[hit["chunk_id"]]
            if any(hit[field] != chunk[field] for field in ("text", "source", "page", "document_id")):
                raise ValueError("Trace no longer matches source chunk")
            expected_metadata = {**chunk["metadata"], "document_id": chunk["document_id"],
                                 "source": chunk["source"], "page": chunk["page"]}
            if hit["metadata"] != expected_metadata:
                raise ValueError("Trace metadata no longer matches the source chunk")
            if not np.isfinite(hit["cosine_distance"]) or not np.isclose(hit["score"], 1 - hit["cosine_distance"]):
                raise ValueError("Invalid native cosine distance/score")
    units = [json.loads(line) for line in (data_dir / "source_units.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rescored = score_traces(traces, benchmark["questions"], report.get("evidence_reviews", []), units)
    if any(old["evaluation"] != new["evaluation"] for old, new in zip(traces, rescored)) or summarize(rescored) != report["metrics"]:
        raise ValueError("Saved evidence labels or metrics need --score-results")
    selection = report["selection"]
    if not selection["frozen"] or selection["experiment_signature"] != _experiment_signature(report):
        raise ValueError("Frozen experiment signature differs")
    if not (datetime.fromisoformat(selection["frozen_at"]) < datetime.fromisoformat(report["test_started_at"])
            <= datetime.fromisoformat(report["test_completed_at"])):
        raise ValueError("Test was not recorded after the configuration freeze")
    winner = min(selection["candidates"], key=lambda candidate: (*candidate["ranking_key"], candidate["strategy"]))
    if winner["strategy"] != selection["selected_strategy"] or selection["test_results_available_when_selected"]:
        raise ValueError("Selection does not match the recorded pre-test decision")
    for candidate in selection["candidates"]:
        metrics = candidate["dev_metrics"]
        expected_key = [-metrics["evidence_hit_at_3"]["fraction"], -metrics["hit_at_1"]["fraction"],
                        candidate["measured_vector_upsert_seconds"], candidate["chunk_count"]]
        if candidate["ranking_key"] != expected_key:
            raise ValueError("Frozen selection ranking is inconsistent")
    if selection["criteria"] != benchmark["policy"]["selection_criteria_in_order"]:
        raise ValueError("Selection criteria differ from the declared benchmark policy")
    test_vectors = np.asarray(report["test_query_embeddings"]["vectors"], dtype=np.float32)
    test_ids = [question["id"] for question in benchmark["questions"] if question["split"] == "test"]
    if (report["test_query_embeddings"]["question_ids"] != test_ids or test_vectors.shape != (len(test_ids), 384)
            or not np.isfinite(test_vectors).all() or not np.allclose(np.linalg.norm(test_vectors, axis=1), 1, atol=1e-5)
            or report["test_query_embeddings"]["embedding_fingerprint"] != fingerprint):
        raise ValueError("Cached test query embeddings are inconsistent")
    samples = {sample["sample_id"]: sample for sample in artifact["coherence_sampling"]["positions"]}
    coherence = report["audits"]["coherence"]["labels"]
    if {(label["sample_id"], label["strategy"]) for label in coherence} != {(sample, strategy) for sample in samples for strategy in STRATEGIES}:
        raise ValueError("Coherence samples differ between strategies")
    for label in coherence:
        sample = samples[label["sample_id"]]
        chunk_id = sample["chunk_id_by_strategy"][label["strategy"]]
        if label["chunk_id"] != chunk_id or label["text_sha256"] != hashlib.sha256(chunks[chunk_id]["text"].encode("utf-8")).hexdigest():
            raise ValueError("Coherence label points to changed chunk text")
    grounding = report["audits"]["grounding"]["labels"]
    test_by_pair = {(trace["question_id"], trace["strategy"]): trace for trace in report["test_traces"]}
    if {(label["question_id"], label["strategy"]) for label in grounding} != set(test_by_pair):
        raise ValueError("Grounding labels do not cover all test traces")
    for label in grounding:
        hits = test_by_pair[(label["question_id"], label["strategy"])]["hits"]
        signature = hashlib.sha256(json.dumps([(hit["chunk_id"], hit["text"]) for hit in hits], ensure_ascii=True).encode("utf-8")).hexdigest()
        if label["top3_signature"] != signature or not set(label["supporting_chunk_ids"]) <= {hit["chunk_id"] for hit in hits}:
            raise ValueError("Grounding label does not match its actual top-3")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_metadata = build_metadata_audit(report["test_traces"], manifest, units, data_dir.parent.parent)
    recorded = report["audits"]["metadata"]
    if ([(label["question_id"], label["strategy"], label["rank"], label["chunk_id"], label["checks"], label["sufficient_and_correct"])
         for label in recorded["labels"]] != [(label["question_id"], label["strategy"], label["rank"], label["chunk_id"], label["checks"], label["sufficient_and_correct"])
                                             for label in expected_metadata["labels"]]
            or recorded["filter_cases"] != expected_metadata["filter_cases"]):
        raise ValueError("Metadata audit does not match the raw source snapshots")
    if summarize_audits(report["audits"]) != report["audit_summary"]:
        raise ValueError("Rubric summary does not match its labels")
    with (data_dir.parent.parent / "artifacts/experiment_log.csv").open(encoding="utf-8", newline="") as source:
        log_rows = list(csv.DictReader(source))
    if len(log_rows) != 3 or {row["strategy"] for row in log_rows} != set(STRATEGIES):
        raise ValueError("Experiment log rows are incomplete")
    for row in log_rows:
        for split in ("dev", "test"):
            if (row[split + "_hit1"] != _metric_display(report["metrics"][row["strategy"]][split]["hit_at_1"])
                    or row[split + "_hit3"] != _metric_display(report["metrics"][row["strategy"]][split]["evidence_hit_at_3"])):
                raise ValueError("Experiment log metrics are stale")
    return {"questions": len(benchmark["questions"]), "strategy_query_traces": len(traces),
            "retrieved_records": sum(len(trace["hits"]) for trace in traces),
            "selected_before_test": selection["selected_strategy"], "freeze_before_test_verified": True,
            "coherence_labels": len(coherence), "grounding_labels": len(grounding),
            "metadata_audit_records": len(recorded["labels"]), "human_author_review_pending": True,
            "post_freeze_dev_metric_revision": any(candidate["dev_metrics"] != report["metrics"][candidate["strategy"]]["dev"]
                                                   for candidate in selection["candidates"])}


def main() -> None:
    """Run or validate one pipeline stage using the saved experiment artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-benchmark", action="store_true",
                       help="Validate saved corpus, split, filters, and gold evidence without retrieval")
    modes.add_argument("--build-chunks", action="store_true",
                       help="Build A/B chunks, alias C to B, and save one artifact without embeddings/retrieval")
    modes.add_argument("--embed-chunks", action="store_true",
                       help="Create real local embeddings for chunks and dev queries")
    modes.add_argument("--validate-embeddings", action="store_true",
                       help="Validate saved vectors and input hashes without loading the model")
    modes.add_argument("--ingest-store", action="store_true",
                       help="Synchronize persistent Chroma, test re-ingestion, and trace dev top-3")
    modes.add_argument("--check-store", action="store_true",
                       help="Reopen the saved Chroma database and verify persistence in this new process")
    modes.add_argument("--score-dev", action="store_true",
                       help="Score stored dev evidence without running test")
    modes.add_argument("--run-test", action="store_true",
                       help="Freeze the dev decision before encoding/retrieving held-out test questions")
    modes.add_argument("--score-results", action="store_true",
                       help="Aggregate saved dev/test metrics and explicit audit labels without retrieval")
    modes.add_argument("--validate-results", action="store_true",
                       help="Validate all saved traces, metrics, freeze ordering and trace-linked audit labels")
    parser.add_argument("--offline", action="store_true",
                        help="Use the previously downloaded pinned model without network access")
    args = parser.parse_args()
    if args.offline and not args.embed_chunks:
        parser.error("--offline is used with --embed-chunks")
    starter_root = Path(__file__).resolve().parents[1]
    if args.validate_benchmark:
        from .evaluation import validate_benchmark

        try:
            summary = validate_benchmark(starter_root / "data")
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise SystemExit(f"Benchmark validation failed: {error}") from error
        print("Benchmark validation OK")
        print(json.dumps(summary, indent=2))
        return
    if args.build_chunks:
        try:
            summary = build_chunk_artifact(starter_root / "data", starter_root / "results")
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise SystemExit(f"Chunking failed: {error}") from error
        print("Chunking OK")
        print(json.dumps(summary, indent=2))
        return
    if args.embed_chunks or args.validate_embeddings:
        try:
            if args.embed_chunks:
                summary = build_embedding_artifact(starter_root / "data", starter_root / "results", offline=args.offline)
            else:
                from .embeddings import validate_embedding_artifact

                summary = validate_embedding_artifact(starter_root / "data", starter_root / "results")
        except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError) as error:
            raise SystemExit(f"Embedding stage failed: {error}") from error
        print("Embeddings OK" if args.embed_chunks else "Embedding validation OK")
        print(json.dumps(summary, indent=2))
        return
    if args.ingest_store or args.check_store:
        try:
            summary = (ingest_store if args.ingest_store else check_store)(starter_root / "data", starter_root / "results")
        except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError) as error:
            raise SystemExit(f"Store stage failed: {error}") from error
        print("Store ingestion OK" if args.ingest_store else "Store persistence OK")
        print(json.dumps(summary, indent=2))
        return
    if args.score_dev or args.run_test:
        try:
            summary = (score_dev if args.score_dev else run_test)(starter_root / "data", starter_root / "results")
        except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError) as error:
            raise SystemExit(f"Benchmark stage failed: {error}") from error
        print("Dev scoring OK" if args.score_dev else "Test evaluation OK")
        print(json.dumps(summary, indent=2))
        return
    if args.score_results or args.validate_results:
        try:
            summary = (score_results if args.score_results else validate_results)(starter_root / "data", starter_root / "results")
        except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError) as error:
            raise SystemExit(f"Result stage failed: {error}") from error
        print("Result scoring OK" if args.score_results else "Benchmark result validation OK")
        print(json.dumps(summary, indent=2))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
