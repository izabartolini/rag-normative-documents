
import csv
import json
import sys

from pathlib import Path
from statistics import mean

from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT_DIR / "src"))

from retrieval import (
    MODEL_NAME,
    load_strategy,
    encode_query,
    retrieve
)


EVALUATION_DIR = ROOT_DIR / "evaluation"
RESULTS_DIR = ROOT_DIR / "results"

MAPPING_PATH = (
    EVALUATION_DIR / "pilot_relevance_mapping.json"
)

TOP_K = 5

STRATEGIES = ["fixed", "hierarchical"]

METRICS = [
    "precision_at_5",
    "recall_at_5",
    "reciprocal_rank_at_5",
    "hit_at_5",
    "device_recall_at_5",
    "retrieved_tokens",
    "retrieval_time_ms"
]


def load_mapping():

    return json.loads(
        MAPPING_PATH.read_text(encoding="utf-8")
    )


def calculate_precision(retrieved_ids, relevant_ids):

    relevant_retrieved = (
        set(retrieved_ids) & set(relevant_ids)
    )

    return len(relevant_retrieved) / TOP_K


def calculate_recall(retrieved_ids, relevant_ids):

    if not relevant_ids:
        raise ValueError(
            "Pergunta sem fragmentos relevantes."
        )

    relevant_retrieved = (
        set(retrieved_ids) & set(relevant_ids)
    )

    return len(relevant_retrieved) / len(
        set(relevant_ids)
    )


def calculate_reciprocal_rank(
    retrieved_ids,
    relevant_ids
):

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1
    ):

        if chunk_id in relevant_ids:
            return 1 / rank

    return 0.0


def calculate_hit(retrieved_ids, relevant_ids):

    return int(
        bool(
            set(retrieved_ids) & set(relevant_ids)
        )
    )


def calculate_device_recall(
    retrieved_ids,
    relevant_devices,
    strategy
):

    if not relevant_devices:
        raise ValueError(
            "Pergunta sem dispositivos relevantes."
        )

    retrieved_set = set(retrieved_ids)

    covered_devices = []
    missing_devices = []

    for device in relevant_devices:

        device_chunks = set(
            device["chunks"][strategy]
        )

        if retrieved_set & device_chunks:

            covered_devices.append(
                device["reference"]
            )

        else:

            missing_devices.append(
                device["reference"]
            )

    recall = (
        len(covered_devices) /
        len(relevant_devices)
    )

    return recall, covered_devices, missing_devices


def validate_mapping(mappings, datasets):

    question_ids = set()

    for question in mappings:

        if question["id"] in question_ids:
            raise ValueError(
                "Identificador de pergunta duplicado."
            )

        question_ids.add(question["id"])

        for strategy in STRATEGIES:

            chunks, _ = datasets[strategy]

            available_ids = {
                chunk["id"] for chunk in chunks
            }

            if len(chunks) < TOP_K:
                raise ValueError(
                    "Quantidade insuficiente de fragmentos."
                )

            relevant_ids = set(
                question["strategies"][strategy][
                    "relevant_chunks"
                ]
            )

            if not relevant_ids:
                raise ValueError(
                    f"{question['id']} sem fragmentos "
                    f"relevantes em {strategy}."
                )

            if not relevant_ids.issubset(available_ids):
                raise ValueError(
                    "Mapeamento contém IDs inexistentes."
                )

            for device in question["relevant_devices"]:

                device_ids = set(
                    device["chunks"][strategy]
                )

                if not device_ids:
                    raise ValueError(
                        "Dispositivo sem correspondência "
                        "integral. Revise o mapeamento."
                    )

                if not device_ids.issubset(relevant_ids):
                    raise ValueError(
                        "Mapeamento de dispositivos "
                        "inconsistente."
                    )


def evaluate_question(
    question,
    strategy,
    query_embedding,
    chunks,
    embeddings
):

    results, retrieval_time = retrieve(
        query_embedding,
        chunks,
        embeddings,
        top_k=TOP_K
    )

    retrieved_ids = [
        result["id"] for result in results
    ]

    relevant_ids = question["strategies"][
        strategy
    ]["relevant_chunks"]

    precision = calculate_precision(
        retrieved_ids,
        relevant_ids
    )

    recall = calculate_recall(
        retrieved_ids,
        relevant_ids
    )

    reciprocal_rank = calculate_reciprocal_rank(
        retrieved_ids,
        relevant_ids
    )

    hit = calculate_hit(
        retrieved_ids,
        relevant_ids
    )

    (
        device_recall,
        covered_devices,
        missing_devices
    ) = calculate_device_recall(
        retrieved_ids,
        question["relevant_devices"],
        strategy
    )

    retrieved_tokens = sum(
        result["token_count"]
        for result in results
    )

    return {
        "question_id": question["id"],
        "question": question["question"],
        "category": question["category"],
        "strategy": strategy,

        "precision_at_5": precision,
        "recall_at_5": recall,
        "reciprocal_rank_at_5": reciprocal_rank,
        "hit_at_5": hit,
        "device_recall_at_5": device_recall,

        "retrieved_tokens": retrieved_tokens,

        "retrieval_time_ms": (
            retrieval_time * 1000
        ),

        "retrieved_chunks": retrieved_ids,
        "relevant_chunks": relevant_ids,

        "covered_devices": covered_devices,
        "missing_devices": missing_devices,

        "results": results
    }


def print_question_result(result):

    print("\nEstratégia:", result["strategy"])

    print(
        "Precision@5:",
        round(result["precision_at_5"], 4)
    )

    print(
        "Recall@5:",
        round(result["recall_at_5"], 4)
    )

    print(
        "RR@5:",
        round(result["reciprocal_rank_at_5"], 4)
    )

    print("Hit@5:", result["hit_at_5"])

    print(
        "Device Recall@5:",
        round(result["device_recall_at_5"], 4)
    )

    print(
        "Tokens recuperados:",
        result["retrieved_tokens"]
    )

    print(
        "Tempo de busca (ms):",
        round(result["retrieval_time_ms"], 4)
    )

    print(
        "Fragmentos recuperados:",
        result["retrieved_chunks"]
    )

    print(
        "Fragmentos relevantes:",
        result["relevant_chunks"]
    )

    print(
        "Dispositivos recuperados:",
        result["covered_devices"]
    )

    print(
        "Dispositivos não recuperados:",
        result["missing_devices"]
    )


def summarize_results(all_results):

    summary = {}

    for strategy in STRATEGIES:

        strategy_results = [
            result for result in all_results
            if result["strategy"] == strategy
        ]

        summary[strategy] = {
            "total_questions": len(strategy_results),

            "mean_precision_at_5": mean(
                r["precision_at_5"]
                for r in strategy_results
            ),

            "mean_recall_at_5": mean(
                r["recall_at_5"]
                for r in strategy_results
            ),

            "mrr_at_5": mean(
                r["reciprocal_rank_at_5"]
                for r in strategy_results
            ),

            "hit_rate_at_5": mean(
                r["hit_at_5"]
                for r in strategy_results
            ),

            "mean_device_recall_at_5": mean(
                r["device_recall_at_5"]
                for r in strategy_results
            ),

            "total_covered_devices": sum(
                len(r["covered_devices"])
                for r in strategy_results
            ),

            "total_expected_devices": sum(
                len(r["covered_devices"]) +
                len(r["missing_devices"])
                for r in strategy_results
            ),

            "mean_retrieved_tokens": mean(
                r["retrieved_tokens"]
                for r in strategy_results
            ),

            "mean_retrieval_time_ms": mean(
                r["retrieval_time_ms"]
                for r in strategy_results
            )
        }

        current = summary[strategy]

        current["micro_device_recall_at_5"] = (
            current["total_covered_devices"] /
            current["total_expected_devices"]
        )

    return summary


def save_results(all_results, summary):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    json_path = RESULTS_DIR / "pilot_results.json"

    csv_path = RESULTS_DIR / "pilot_metrics.csv"

    summary_path = (
        RESULTS_DIR / "pilot_summary.json"
    )

    json_path.write_text(
        json.dumps(
            all_results,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    fields = [
        "question_id",
        "question",
        "category",
        "strategy",
        *METRICS
    ]

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore",
            delimiter=";"
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\nArquivos gerados:")

    print(json_path)
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":

    mappings = load_mapping()

    model = SentenceTransformer(MODEL_NAME)

    datasets = {
        strategy: load_strategy(strategy)
        for strategy in STRATEGIES
    }

    validate_mapping(mappings, datasets)

    all_results = []

    print("Modelo:", MODEL_NAME)
    print("Perguntas:", len(mappings))
    print("Top-K:", TOP_K)

    for question in mappings:

        print("\n" + "=" * 60)
        print("ID:", question["id"])
        print("Pergunta:", question["question"])
        print("=" * 60)

        query_embedding, query_time = encode_query(
            model,
            question["question"]
        )

        for strategy in STRATEGIES:

            chunks, embeddings = datasets[strategy]

            result = evaluate_question(
                question,
                strategy,
                query_embedding,
                chunks,
                embeddings
            )

            result["query_embedding_time_seconds"] = (
                query_time
            )

            all_results.append(result)

            print_question_result(result)

    summary = summarize_results(all_results)

    print("\n" + "=" * 60)
    print("RESULTADOS GERAIS")
    print("=" * 60)

    for strategy, metrics in summary.items():

        print("\nEstratégia:", strategy.upper())

        for metric, value in metrics.items():

            print(
                f"{metric}:",
                round(value, 6)
            )

    save_results(all_results, summary)
