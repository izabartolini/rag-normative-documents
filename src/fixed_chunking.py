
import json
from pathlib import Path
from transformers import AutoTokenizer

MODEL_NAME = "intfloat/multilingual-e5-small"

CHUNK_SIZE = 400
OVERLAP = 50
MAX_TOKENS = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def count_tokens(text):
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False
        )
    )


def fixed_chunking(text, document_name):
    if not 0 <= OVERLAP < CHUNK_SIZE:
        raise ValueError("Overlap inválido")

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        verbose=False
    )

    tokens = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    chunks = []
    start = 0

    while start < len(tokens):

        end = min(start + CHUNK_SIZE, len(tokens))

        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]

        chunk_text = text[char_start:char_end]

        total_tokens = count_tokens(
            f"passage: {chunk_text}"
        )

        if total_tokens > MAX_TOKENS:
            raise ValueError(
                "Fragmento acima do limite do modelo"
            )

        chunks.append({
            "id": f"fixed_{len(chunks) + 1:04d}",
            "document": document_name,
            "strategy": "fixed",
            "text": chunk_text,
            "token_count": total_tokens,
            "start_token": start,
            "end_token": end,
            "start_char": char_start,
            "end_char": char_end
        })

        if end == len(tokens):
            break

        start = end - OVERLAP

    return chunks


if __name__ == "__main__":

    elements = json.loads(
        Path("data/processed/lgpd_hierarchy.json")
        .read_text(encoding="utf-8")
    )

    text = "\n".join(
        element["text"]
        for element in elements
    )

    chunks = fixed_chunking(
        text,
        "Lei nº 13.709/2018 - LGPD"
    )

    output = Path(
        "data/processed/lgpd_fixed_chunks.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.write_text(
        json.dumps(
            chunks,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    counts = [
        chunk["token_count"]
        for chunk in chunks
    ]

    print("=" * 60)
    print("FRAGMENTAÇÃO POR TAMANHO FIXO")
    print("=" * 60)

    print("Tamanho-alvo:", CHUNK_SIZE)
    print("Overlap:", OVERLAP)
    print("Total de fragmentos:", len(chunks))

    if counts:
        print("Mínimo de tokens:", min(counts))
        print("Máximo de tokens:", max(counts))
        print(
            "Média de tokens:",
            round(sum(counts) / len(counts), 2)
        )

        print(
            "Fragmentos acima do limite:",
            sum(c > MAX_TOKENS for c in counts)
        )

    print("Arquivo gerado:", output)

    for chunk in chunks[:3]:
        print("\nID:", chunk["id"])
        print("Tokens:", chunk["token_count"])
        print("Conteúdo:", chunk["text"])
        print("-" * 60)
