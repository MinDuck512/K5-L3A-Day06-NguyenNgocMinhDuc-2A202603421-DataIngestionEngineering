"""Validate gold data, score sufficient single-chunk evidence, and aggregate audits."""
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


def validate_benchmark(data_dir: Path) -> dict[str, Any]:
    """Check source integrity, split, filters, and literal gold evidence offline.

    This validates authored benchmark data, not retrieval quality or human audit
    labels. Equivalent evidence still needs a documented semantic review.
    """
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    benchmark = json.loads((data_dir / "benchmark.json").read_text(encoding="utf-8"))
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    units_path = data_dir / "source_units.jsonl"
    units = [json.loads(line) for line in units_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for key, path in (("corpus_units_sha256", units_path), ("corpus_manifest_sha256", manifest_path)):
        require(benchmark.get(key) == hashlib.sha256(path.read_bytes()).hexdigest(),
                f"Frozen corpus changed: {path.name}")
    require(len(units) >= 30, "Corpus needs at least 30 valid source units")
    require(len(units) == manifest["total_valid_source_units"], "Manifest unit count differs")
    require(len({unit["unit_id"] for unit in units}) == len(units), "Duplicate source unit IDs")
    require(len({unit["text"] for unit in units}) == len(units), "Duplicated source content")
    by_key = {(unit["document_id"], unit["page"]): unit for unit in units}
    require(len(by_key) == len(units), "Duplicate document/page positions")
    source_ids = {source["source_id"] for source in manifest["sources"]}
    require({unit["document_id"] for unit in units} == source_ids, "Unlisted or missing document")
    lab_root = data_dir.parent.parent
    for source in manifest["sources"]:
        raw = (lab_root / source["local_source_path"]).read_bytes()
        text = (lab_root / source["normalized_text_path"]).read_text(encoding="utf-8")
        require(hashlib.sha256(raw).hexdigest() == source["raw_sha256"], "Raw source hash differs")
        require(hashlib.sha256(text.encode("utf-8")).hexdigest() == source["normalized_sha256"],
                "Normalized source hash differs")
        document_units = sorted((unit for unit in units if unit["document_id"] == source["source_id"]),
                                key=lambda unit: unit["page"])
        require([unit["unit_id"] for unit in document_units] == source["page_ids"],
                "Manifest page IDs differ")
        previous_end = 0
        for number, unit in enumerate(document_units, start=1):
            start, end = unit["char_start"], unit["char_end"]
            require(unit["page"] == number, "Source unit numbers must be consecutive")
            require(previous_end <= start < end <= len(text), "Invalid or overlapping source boundaries")
            require(not text[previous_end:start].strip(), "Useful text omitted between source units")
            require(text[start:end] == unit["text"], "Source text does not match saved boundaries")
            require(unit["char_count"] == len(unit["text"]) >= 1500, "Source unit too short")
            require(unit["metadata"] == source["metadata"], "Source metadata differs")
            require(unit["source"] == source["source_path_or_url"], "Source URL differs")
            previous_end = end
        require(not text[previous_end:].strip(), "Useful source text omitted at document end")
    license_info = manifest["license_snapshot"]
    require(hashlib.sha256((lab_root / license_info["path"]).read_bytes()).hexdigest()
            == license_info["sha256"], "License snapshot hash differs")

    questions = benchmark["questions"]
    require(len(questions) >= 20, "Benchmark needs at least 20 questions")
    require(len({question["id"] for question in questions}) == len(questions), "Duplicate question IDs")
    require(len({question["question"] for question in questions}) == len(questions), "Duplicate questions")
    splits = Counter(question["split"] for question in questions)
    require(set(splits) == {"dev", "test"} and splits["dev"] >= 15 and splits["test"] >= 5,
            "Need at least 15 dev / 5 test questions")
    filtered = Counter()
    evidence_count = 0
    alternative_count = 0
    documents_by_split: dict[str, set[str]] = {"dev": set(), "test": set()}
    for question in questions:
        qid = question["id"]
        for key in ("question", "expected_answer", "evidence_quote"):
            require(isinstance(question.get(key), str) and bool(question[key].strip()),
                    f"{qid}: missing {key}")
        primary = by_key.get((question["gold_source_id"], question["gold_page"]))
        require(primary is not None, f"{qid}: unknown source/page")
        require(question["gold_unit_id"] == primary["unit_id"], f"{qid}: wrong gold unit ID")
        require(question["gold_source_url"] == primary["source"], f"{qid}: wrong gold URL")
        start, end = question["evidence_char_start"], question["evidence_char_end"]
        require(0 <= start < end <= len(primary["text"]), f"{qid}: invalid evidence boundaries")
        require(primary["text"][start:end] == question["evidence_quote"], f"{qid}: evidence does not match")
        golds = [question, *question.get("alternative_gold", [])]
        filters = question.get("filter")
        if filters is not None:
            require(isinstance(filters, dict) and set(filters) == {"chapter"},
                    f"{qid}: only explicit chapter filters are permitted")
            require(type(filters["chapter"]) is int, f"{qid}: chapter must be an integer")
            provided_chapters = re.findall(r"\bchương\s+(\d+)\b", question["question"], flags=re.IGNORECASE)
            require(str(filters["chapter"]) in provided_chapters, f"{qid}: filter not provided in question")
            require(question.get("filter_origin") == "explicit_question_context", f"{qid}: undocumented filter")
            filtered[question["split"]] += 1
        for gold in golds:
            unit = by_key.get((gold["gold_source_id"], gold["gold_page"]))
            require(unit is not None, f"{qid}: invalid alternative source/page")
            require(unit["unit_id"] == gold["gold_unit_id"], f"{qid}: incorrect evidence unit ID")
            require(bool(gold["evidence_quote"]) and gold["evidence_quote"] in unit["text"],
                    f"{qid}: evidence missing from source")
            require(not filters or all(unit["metadata"].get(key) == value for key, value in filters.items()),
                    f"{qid}: gold is outside the supplied filter")
            documents_by_split[question["split"]].add(gold["gold_source_id"])
        chapter = primary["metadata"]["chapter"]
        require(chapter in benchmark["policy"][question["split"] + "_chapters"],
                f"{qid}: chapter does not match the declared split")
        evidence_count += 1
        alternative_count += len(golds) - 1
    require(sum(filtered.values()) >= 3 and filtered["test"] >= 1,
            "Need at least 3 filter questions, including at least 1 test question")
    overlap = documents_by_split["dev"] & documents_by_split["test"]
    require(not overlap, "Document-held-out split has shared gold documents")
    require(benchmark["policy"].get("fixed_before_first_retrieval") is True, "Split must be fixed before retrieval")
    require(bool(benchmark["policy"].get("selection_criteria_in_order")), "Missing strategy selection criteria")
    return {
        "questions": len(questions), "dev": splits["dev"], "test": splits["test"],
        "filter_questions": sum(filtered.values()), "test_filter_questions": filtered["test"],
        "gold_evidence_verified": evidence_count, "alternative_evidence_verified": alternative_count,
        "corpus_units": len(units), "corpus_snapshot": "OK",
        "dev_test_document_overlap": len(overlap), "validation_only": True,
    }


def _evidence_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]],
                   evidence_reviews: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Score one sufficient chunk, with explicit source-checked semantic reviews.

    Reviews are keyed by question/chunk, shared across all strategies. They do
    not supply coherence/grounding labels. A union across chunks is diagnostic
    only and never replaces the primary single-chunk definition.
    """
    golds = [question, *question.get("alternative_gold", [])]
    reviews = {review["chunk_id"]: review for review in (evidence_reviews or [])
               if review["question_id"] == question["id"]}
    labels = []
    for index, hit in enumerate(hits[:3], start=1):
        literal = any(
            (hit["document_id"], hit["page"]) == (gold["gold_source_id"], gold["gold_page"])
            and _evidence_text(gold["evidence_quote"]) in _evidence_text(hit["text"])
            for gold in golds
        )
        label = {"rank": index, "chunk_id": hit["chunk_id"], "literal_gold_hit": literal,
                 "sufficient_evidence": literal,
                 "method": "source_unit_and_complete_gold_quote" if literal else "no_complete_literal_gold"}
        if hit["chunk_id"] in reviews:
            review = reviews[hit["chunk_id"]]
            if type(review.get("sufficient")) is not bool or not review.get("reason") or not review.get("reviewer"):
                raise ValueError("Semantic evidence review needs a boolean, reason, and named reviewer")
            if (review["document_id"], review["page"]) != (hit["document_id"], hit["page"]):
                raise ValueError("Semantic review points to a different source")
            if review["sufficient"]:
                if not review.get("supporting_quote") or _evidence_text(review["supporting_quote"]) not in _evidence_text(hit["text"]):
                    raise ValueError("Reviewed supporting quote is missing from the retrieved chunk")
                if question.get("filter") and any(hit["metadata"].get(key) != value for key, value in question["filter"].items()):
                    raise ValueError("Reviewed evidence lies outside the question's requested scope")
            label.update({"sufficient_evidence": review["sufficient"], "method": "documented_semantic_review",
                          "reviewer": review["reviewer"], "reason": review["reason"]})
        labels.append(label)

    start, end = question["evidence_char_start"], question["evidence_char_end"]
    covered_until = start
    intervals = sorted((max(start, hit["metadata"]["unit_char_start"]),
                        min(end, hit["metadata"]["unit_char_end"])) for hit in hits[:3]
                       if (hit["document_id"], hit["page"]) == (question["gold_source_id"], question["gold_page"]))
    for left, right in intervals:
        if left <= covered_until and right > covered_until:
            covered_until = right
    hit1 = bool(labels and labels[0]["sufficient_evidence"])
    hit3 = any(label["sufficient_evidence"] for label in labels)
    return {
        "question_id": question["id"], "split": question["split"],
        "expected_answer": question["expected_answer"],
        "gold_source_id": question["gold_source_id"], "gold_page": question["gold_page"],
        "gold_unit_id": question["gold_unit_id"], "evidence_quote": question["evidence_quote"],
        "hit_at_1": hit1, "evidence_hit_at_3": hit3,
        "literal_hit_at_1": bool(labels and labels[0]["literal_gold_hit"]),
        "literal_hit_at_3": any(label["literal_gold_hit"] for label in labels),
        "primary_gold_covered_by_multiple_top3_chunks": covered_until >= end and not any(label["literal_gold_hit"] for label in labels),
        "per_hit_labels": labels,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep numerators/denominators and literal-only baselines for every split."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for trace in traces:
        groups.setdefault((trace["strategy"], trace["split"]), []).append(trace["evaluation"])
    summary: dict[str, Any] = {}
    for (strategy, split), evaluations in sorted(groups.items()):
        total = len(evaluations)
        row = {"questions": total}
        for field in ("hit_at_1", "evidence_hit_at_3", "literal_hit_at_1", "literal_hit_at_3"):
            count = sum(bool(evaluation[field]) for evaluation in evaluations)
            row[field] = {"hits": count, "total": total, "fraction": count / total}
        row["primary_gold_split_across_top3"] = sum(
            evaluation["primary_gold_covered_by_multiple_top3_chunks"] for evaluation in evaluations)
        summary.setdefault(strategy, {})[split] = row
    return summary


def score_traces(traces: list[dict], questions: list[dict], reviews: list[dict], units: list[dict]) -> list[dict]:
    """Verify reviewed evidence against the frozen source before using its label."""
    question_by_id = {question["id"]: question for question in questions}
    unit_by_key = {(unit["document_id"], unit["page"]): unit for unit in units}
    pairs = [(review["question_id"], review["chunk_id"]) for review in reviews]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate semantic review for the same question/chunk")
    for review in reviews:
        if review["question_id"] not in question_by_id:
            raise ValueError("Review references an unknown benchmark question")
        source = unit_by_key[(review["document_id"], review["page"])]
        if review.get("supporting_quote") and _evidence_text(review["supporting_quote"]) not in _evidence_text(source["text"]):
            raise ValueError("Semantic review quote does not exist in its declared source unit")
    scored = []
    for trace in traces:
        question = question_by_id[trace["question_id"]]
        if trace["split"] != question["split"]:
            raise ValueError("Trace split differs from the frozen benchmark")
        scored.append({**trace, "evaluation": evaluate_query(question, trace["hits"], reviews)})
    return scored


def build_metadata_audit(test_traces: list[dict], manifest: dict, units: list[dict], lab_root: Path) -> dict:
    """Check every test result against actual HTML headings, source units and offsets."""
    from .ingestion import MainTextParser

    sources = {}
    for source in manifest["sources"]:
        parser = MainTextParser()
        parser.feed((lab_root / source["local_source_path"]).read_text(encoding="utf-8"))
        parser.close()
        title = "".join(parser.heading_parts).strip()
        version = re.search(r"Python\s+(\d+\.\d+\.\d+[^\s]*)\s+documentation", "".join(parser.title_parts)).group(1)
        chapter = int(re.match(r"(\d+)\.", title).group(1))
        sources[source["source_id"]] = {"source": source, "text": parser.text(),
                                         "title": title, "version": version,
                                         "chapter": chapter, "language": parser.language}
    by_key = {(unit["document_id"], unit["page"]): unit for unit in units}
    labels, filter_cases = [], []
    reviewer = "Codex (source checks plus AI inspection; human author review pending)"
    for trace in test_traces:
        if trace["split"] != "test":
            raise ValueError("Metadata audit is defined on the held-out test top-3")
        for hit in trace["hits"]:
            unit = by_key[(hit["document_id"], hit["page"])]
            source = sources[hit["document_id"]]
            metadata = hit["metadata"]
            required = ("document_id", "source", "page", "chapter", "version", "language", "unit_id")
            checks = {
                "required_fields_present": all(key in metadata for key in required),
                "document_id": metadata.get("document_id") == unit["document_id"],
                "source": metadata.get("source") == hit["source"] == source["source"]["source_path_or_url"],
                "page_and_unit": metadata.get("page") == hit["page"] == unit["page"] and metadata.get("unit_id") == unit["unit_id"],
                "chapter_from_html_heading": metadata.get("chapter") == source["chapter"],
                "chapter_title": metadata.get("chapter_title") == source["title"],
                "version_from_html_title": metadata.get("version") == source["version"],
                "language_from_html": metadata.get("language") == source["language"],
                "source_unit_boundaries": source["text"][unit["char_start"]:unit["char_end"]] == unit["text"],
                "chunk_boundaries": unit["text"][metadata["unit_char_start"]:metadata["unit_char_end"]] == hit["text"],
                "document_boundaries": source["text"][metadata["document_char_start"]:metadata["document_char_end"]] == hit["text"],
            }
            labels.append({
                "question_id": trace["question_id"], "strategy": trace["strategy"], "rank": hit["rank"],
                "chunk_id": hit["chunk_id"], "document_id": hit["document_id"], "page": hit["page"],
                "source_heading": source["title"], "checks": checks,
                "sufficient_and_correct": all(checks.values()),
                "reason": "Đối chiếu URL, heading/chương, version, ngôn ngữ và ranh giới text với snapshot HTML và đơn vị nguồn; "
                          + ("các trường đều khớp." if all(checks.values()) else "có trường không khớp: " + ", ".join(key for key, value in checks.items() if not value)),
                "reviewer": reviewer, "student_review": "pending",
            })
        requested = trace.get("requested_filter")
        if requested:
            filter_cases.append({
                "question_id": trace["question_id"], "strategy": trace["strategy"],
                "requested_filter": requested,
                "actual_chapters": [hit["metadata"]["chapter"] for hit in trace["hits"]],
                "top3_compliant": bool(trace["hits"]) and all(
                    all(hit["metadata"].get(key) == value for key, value in requested.items()) for hit in trace["hits"]),
            })
    return {
        "review_type": "Automated source equality checks plus Codex inspection; not a human author audit",
        "method": "All five test questions, all top-3 results for each strategy; HTML headings/version/language independently parsed from the saved raw snapshots",
        "labels": labels, "filter_cases": filter_cases,
        "usefulness": "source/page/unit offsets giúp mở đúng đoạn nguồn; chapter giúp giới hạn phạm vi được yêu cầu, nhưng không bảo đảm tìm đúng evidence bên trong chương. version/language cố định trong corpus này nên ít giúp phân biệt kết quả.",
    }


def summarize_audits(audits: dict) -> dict:
    """Aggregate actual rubric labels; never infer them from retrieval Hit@3."""
    result = {}
    for strategy in ("A", "B", "C"):
        coherence = [label for label in audits["coherence"]["labels"] if label["strategy"] == strategy]
        grounding = [label for label in audits["grounding"]["labels"] if label["strategy"] == strategy]
        metadata = [label for label in audits["metadata"]["labels"] if label["strategy"] == strategy]
        filters = [case for case in audits["metadata"]["filter_cases"] if case["strategy"] == strategy]
        if (len(coherence), len(grounding), len(metadata)) != (10, 5, 15):
            raise ValueError("Audit needs 10 coherence positions, 5 test grounding labels, and 15 metadata records per strategy")
        for label in coherence + grounding:
            if type(label.get("score")) is not int or label["score"] not in (0, 1, 2) or not label.get("reason") or not label.get("reviewer"):
                raise ValueError("Rubric labels need a valid 0/1/2 score, reason and reviewer")
        result[strategy] = {
            "coherence": {"mean_out_of_2": sum(label["score"] for label in coherence) / len(coherence),
                          "sample_count": len(coherence)},
            "metadata": {"correct_and_sufficient": sum(label["sufficient_and_correct"] for label in metadata),
                         "checked_records": len(metadata)},
            "filter_compliance": {"compliant_questions": sum(case["top3_compliant"] for case in filters),
                                  "filter_questions": len(filters), "display": "N/A" if not filters else "n/N"},
            "grounding": {"counts_0_1_2": {str(score): sum(label["score"] == score for label in grounding) for score in (0, 1, 2)},
                          "mean_out_of_2": sum(label["score"] for label in grounding) / len(grounding),
                          "questions": len(grounding)},
            "review_type": "AI-assisted proposed labels; human author review pending",
        }
    return result
