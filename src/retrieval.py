
import json
import time
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"

PROCESSED_DIR = Path("data/processed")
EMBEDDINGS_DIR = Path("data/embeddings")
RESULTS_DIR = Path("results")

TOP_K = 5


def load_strategy(strategy):

    chunks_path = (
        PROCESSED_DIR /
        f"lgpd_{strategy}_chunks.json"
    )

    embeddings_path = (
        EMBEDDINGS_DIR /
        f"lgpd_{strategy}_embeddings.npy"
    )

    chunks = json.loads(
        chunks_path.read_text(encoding="utf-8")
    )

    embeddings = np.load(
        embeddings_path,
        allow_pickle=False
    )

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Quantidade de fragmentos e "
            "embeddings incompatível."
        )

    if embeddings.ndim != 2:
        raise ValueError("Matriz inválida.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings inválidos.")

    if not np.allclose(
        np.linalg.norm(embeddings, axis=1),
        1.0,
        atol=1e-5
    ):
        raise ValueError(
            "Embeddings não normalizados."
        )

    return chunks, embeddings


def encode_query(model, question):

    text = f"query: {question}"

    tokens = model.tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=False
    )

    if len(tokens) > model.max_seq_length:
        raise ValueError(
            "Pergunta acima do limite do modelo."
        )

    start = time.perf_counter()

    embedding = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    elapsed = time.perf_counter() - start

    return embedding, elapsed


def retrieve(
    query_embedding,
    chunks,
    embeddings,
    top_k=TOP_K
):

    if embeddings.shape[1] != len(query_embedding):
        raise ValueError(
            "Dimensões vetoriais incompatíveis."
        )

    start = time.perf_counter()

    similarities = embeddings @ query_embedding

    indices = np.argsort(
        -similarities,
        kind="stable"
    )[:top_k]

    elapsed = time.perf_counter() - start

    results = []

    for rank, index in enumerate(indices, start=1):

        chunk = chunks[index]

        results.append({
            "rank": rank,
            "id": chunk["id"],
            "similarity": float(
                similarities[index]
            ),
            "token_count": chunk["token_count"],
            "path": chunk.get("path"),
            "element_paths": chunk.get(
                "element_paths", []
            ),
            "text": chunk["text"]
        })

    return results, elapsed


def print_results(strategy, results, elapsed):

    print("\n" + "=" * 60)
    print("ESTRATÉGIA:", strategy.upper())
    print("=" * 60)

    print(
        "Tempo de busca:",
        round(elapsed * 1000, 4),
        "ms"
    )

    print(
        "Total de tokens recuperados:",
        sum(r["token_count"] for r in results)
    )

    for result in results:

        print("\nPosição:", result["rank"])
        print("ID:", result["id"])

        print(
            "Similaridade:",
            round(result["similarity"], 6)
        )

        print("Tokens:", result["token_count"])

        if result["path"]:
            print("Caminho:", result["path"])

        print("Conteúdo:")
        print(result["text"])
        print("-" * 60)


if __name__ == "__main__":

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model = SentenceTransformer(MODEL_NAME)

    question = "O que são dados pessoais sensíveis?"

    print("Modelo:", MODEL_NAME)
    print("Pergunta:", question)
    print("Top-K:", TOP_K)

    query_embedding, query_time = encode_query(
        model,
        question
    )

    print(
        "Tempo de geração do embedding da pergunta:",
        round(query_time, 4),
        "segundos"
    )

    experiment = {
        "model": MODEL_NAME,
        "question": question,
        "top_k": TOP_K,
        "query_embedding_time_seconds": query_time,
        "strategies": {}
    }

    for strategy in ["fixed", "hierarchical"]:

        chunks, embeddings = load_strategy(
            strategy
        )

        results, elapsed = retrieve(
            query_embedding,
            chunks,
            embeddings
        )

        print_results(
            strategy,
            results,
            elapsed
        )

        experiment["strategies"][strategy] = {
            "retrieval_time_seconds": elapsed,
            "total_retrieved_tokens": sum(
                r["token_count"]
                for r in results
            ),
            "results": results
        }

    output_path = (
        RESULTS_DIR /
        "lgpd_retrieval_test.json"
    )

    output_path.write_text(
        json.dumps(
            experiment,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    print("\nArquivo gerado:", output_path)
