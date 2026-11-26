import json
from pathlib import Path
from collections import defaultdict
from statistics import mean, median
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"

model = SentenceTransformer(MODEL_NAME)
tokenizer = model.tokenizer

MAX_TOKENS = model.max_seq_length

elements = json.loads(
    Path("data/processed/lgpd_hierarchy.json")
    .read_text(encoding="utf-8")
)

def count_tokens(text):
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False
        )
    )


statistics = defaultdict(list)
oversized = []
content_counts = []
complete_counts = []
path_counts = []

for element in elements:

    content = element["text"]
    path = element["path"]

    content_tokens = count_tokens(
        f"passage: {content}"
    )

    complete_text = (
        f"passage: {path}\n{content}"
    )

    complete_tokens = count_tokens(complete_text)

    path_tokens = complete_tokens - content_tokens

    content_counts.append(content_tokens)
    complete_counts.append(complete_tokens)
    path_counts.append(path_tokens)

    statistics[element["type"]].append(
        complete_tokens
    )

    if complete_tokens > MAX_TOKENS:
        oversized.append({
            "path": path,
            "tokens": complete_tokens
        })


print("=" * 60)
print("ANÁLISE DE TOKENS")
print("=" * 60)

print("Modelo:", MODEL_NAME)
print("Limite:", MAX_TOKENS)
print("Total de elementos:", len(elements))

print("\nCONTEÚDO SEM CAMINHO")
print("Mínimo:", min(content_counts))
print("Máximo:", max(content_counts))
print("Média:", round(mean(content_counts), 2))
print("Mediana:", median(content_counts))

print("\nCONTEÚDO COM CAMINHO")
print("Mínimo:", min(complete_counts))
print("Máximo:", max(complete_counts))
print("Média:", round(mean(complete_counts), 2))
print("Mediana:", median(complete_counts))

print("\nCUSTO ADICIONAL DO CAMINHO")
print("Média:", round(mean(path_counts), 2))
print("Máximo:", max(path_counts))

print("\nDISTRIBUIÇÃO POR TIPO")

for element_type, values in statistics.items():
    print(f"\nTipo: {element_type}")
    print("Quantidade:", len(values))
    print("Mínimo:", min(values))
    print("Máximo:", max(values))
    print("Média:", round(mean(values), 2))
    print("Mediana:", median(values))

print("\nELEMENTOS ACIMA DO LIMITE")
print("Quantidade:", len(oversized))

for element in oversized:
    print("\nCaminho:", element["path"])
    print("Tokens:", element["tokens"])
