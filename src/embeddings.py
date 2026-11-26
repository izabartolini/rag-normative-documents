
import json
import time
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"

PROCESSED_DIR = Path("data/processed")
EMBEDDINGS_DIR = Path("data/embeddings")

BATCH_SIZE = 16


def load_chunks(file_path):

    with open(file_path, encoding="utf-8") as file:
        return json.load(file)


def generate_embeddings(model, chunks):

    texts = [
        f"passage: {chunk['text']}"
        for chunk in chunks
    ]

    for chunk, text in zip(chunks, texts):

        tokens = model.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
            verbose=False
        )

        if len(tokens) > model.max_seq_length:
            raise ValueError(
                f"Fragmento acima do limite: "
                f"{chunk['id']}"
            )

    start = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    elapsed = time.perf_counter() - start

    return embeddings, elapsed


def validate_embeddings(embeddings, chunks):

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Quantidade de vetores incompatível."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "Formato da matriz inválido."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Foram encontrados valores inválidos."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError(
            "Os vetores não estão normalizados."
        )

    return norms


def process_strategy(model, strategy):

    input_path = (
        PROCESSED_DIR /
        f"lgpd_{strategy}_chunks.json"
    )

    output_path = (
        EMBEDDINGS_DIR /
        f"lgpd_{strategy}_embeddings.npy"
    )

    chunks = load_chunks(input_path)

    print("\n" + "=" * 60)
    print("ESTRATÉGIA:", strategy.upper())
    print("=" * 60)

    embeddings, elapsed = generate_embeddings(
        model,
        chunks
    )

    norms = validate_embeddings(
        embeddings,
        chunks
    )

    np.save(output_path, embeddings)

    print("Total de fragmentos:", len(chunks))
    print("Total de embeddings:", len(embeddings))
    print("Formato da matriz:", embeddings.shape)
    print("Dimensão:", embeddings.shape[1])
    print("Tipo dos dados:", embeddings.dtype)
    print("Norma mínima:", norms.min())
    print("Norma máxima:", norms.max())
    print("Valores inválidos:", 0)

    print(
        "Tempo de geração:",
        round(elapsed, 4),
        "segundos"
    )

    print(
        "Tempo médio por fragmento:",
        round(elapsed / len(chunks), 4),
        "segundos"
    )

    print(
        "Tamanho do arquivo:",
        round(output_path.stat().st_size / 1024, 2),
        "KB"
    )

    print("Arquivo gerado:", output_path)


if __name__ == "__main__":

    EMBEDDINGS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model = SentenceTransformer(MODEL_NAME)

    print("Modelo:", MODEL_NAME)
    print("Limite de tokens:", model.max_seq_length)
    print("Dispositivo:", model.device)

    process_strategy(model, "fixed")
    process_strategy(model, "hierarchical")
