
import json
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer

MODEL_NAME = "intfloat/multilingual-e5-small"

CHUNK_SIZE = 400
MAX_TOKENS = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def count_tokens(text, special_tokens=True):
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=special_tokens,
            truncation=False,
            verbose=False
        )
    )


def get_article_path(path):
    parts = path.split(" > ")

    for i, part in enumerate(parts):
        if part.startswith("Art."):
            return " > ".join(parts[:i + 1])

    return None


def group_by_article(elements):
    articles = defaultdict(list)

    for element in elements:
        article_path = get_article_path(
            element["path"]
        )

        if article_path:
            articles[article_path].append(element)

    return articles


def hierarchical_chunking(elements, document_name):

    articles = group_by_article(elements)

    elements_by_path = {
        element["path"]: element
        for element in elements
    }

    chunks = []

    def build_content(group):

        lines = []
        included = set()

        for element in group:

            parts = element["path"].split(" > ")

            # Recupera o contexto dos elementos
            # superiores ao dispositivo atual.
            for i in range(1, len(parts)):

                parent_path = " > ".join(parts[:i])

                parent = elements_by_path.get(
                    parent_path
                )

                if parent is None:
                    continue

                if parent["type"] in (
                    "capitulo", "secao"
                ):
                    continue

                if parent_path not in included:
                    lines.append(parent["text"])
                    included.add(parent_path)

            # Adiciona o próprio dispositivo.
            if element["path"] not in included:
                lines.append(element["text"])
                included.add(element["path"])

        return "\n".join(lines)

    def build_text(article_path, group):

        content = build_content(group)

        return f"{article_path}\n{content}"

    def fits(article_path, group):

        content = build_content(group)
        text = build_text(article_path, group)

        content_tokens = count_tokens(
            content,
            special_tokens=False
        )

        total_tokens = count_tokens(
            f"passage: {text}"
        )

        return (
            content_tokens <= CHUNK_SIZE
            and total_tokens <= MAX_TOKENS
        )

    def save_chunk(article_path, group):

        if not group:
            return

        text = build_text(article_path, group)

        total_tokens = count_tokens(
            f"passage: {text}"
        )

        if total_tokens > MAX_TOKENS:
            raise ValueError(
                f"Fragmento acima do limite: "
                f"{article_path}"
            )

        chunks.append({
            "id": f"hierarchical_{len(chunks) + 1:04d}",
            "document": document_name,
            "strategy": "hierarchical",
            "path": article_path,
            "text": text,
            "token_count": total_tokens,
            "element_paths": [
                element["path"]
                for element in group
            ]
        })

    for article_path, article_elements in articles.items():

        # Caso 1: artigo completo cabe no limite.
        if fits(article_path, article_elements):

            save_chunk(
                article_path,
                article_elements
            )

            continue

        # Caso 2: artigo precisa ser dividido.
        current_group = []

        for element in article_elements:

            candidate = current_group + [element]

            if fits(article_path, candidate):

                current_group = candidate

            else:

                save_chunk(
                    article_path,
                    current_group
                )

                current_group = [element]

                if not fits(
                    article_path,
                    current_group
                ):
                    raise ValueError(
                        "Dispositivo e contexto "
                        "acima do orçamento: "
                        + element["path"]
                    )

        save_chunk(
            article_path,
            current_group
        )

    return chunks


if __name__ == "__main__":

    elements = json.loads(
        Path("data/processed/lgpd_hierarchy.json")
        .read_text(encoding="utf-8")
    )

    chunks = hierarchical_chunking(
        elements,
        "Lei nº 13.709/2018 - LGPD"
    )

    output = Path(
        "data/processed/lgpd_hierarchical_chunks.json"
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

    articles = group_by_article(elements)

    expected_paths = {
        element["path"]
        for group in articles.values()
        for element in group
    }

    covered_paths = {
        path
        for chunk in chunks
        for path in chunk["element_paths"]
    }

    missing = expected_paths - covered_paths

    print("=" * 60)
    print("FRAGMENTAÇÃO HIERÁRQUICA")
    print("=" * 60)

    print("Total de artigos:", len(articles))
    print("Total de fragmentos:", len(chunks))

    if counts:
        print("Mínimo de tokens:", min(counts))
        print("Máximo de tokens:", max(counts))
        print(
            "Média de tokens:",
            round(sum(counts) / len(counts), 2)
        )
        print("Total de tokens:", sum(counts))
        print(
            "Fragmentos acima do limite:",
            sum(c > MAX_TOKENS for c in counts)
        )

    print("Dispositivos esperados:", len(expected_paths))
    print("Dispositivos cobertos:", len(covered_paths))
    print("Dispositivos ausentes:", len(missing))

    print("Arquivo gerado:", output)

    for chunk in chunks[:5]:

        print("\nID:", chunk["id"])
        print("Tokens:", chunk["token_count"])
        print("Caminho:", chunk["path"])
        print("Conteúdo:", chunk["text"])
        print("-" * 60)
