
import json
import re
import unicodedata

from pathlib import Path


PROCESSED_DIR = Path("data/processed")
EVALUATION_DIR = Path("evaluation")

QUESTIONS_PATH = (
    EVALUATION_DIR / "pilot_questions.json"
)

HIERARCHY_PATH = (
    PROCESSED_DIR / "lgpd_hierarchy.json"
)

OUTPUT_PATH = (
    EVALUATION_DIR / "pilot_relevance_mapping.json"
)

STRATEGIES = ["fixed", "hierarchical"]


def load_json(path):

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def normalize_text(text):

    text = unicodedata.normalize("NFC", text)
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def find_device(reference, elements):

    normalized_reference = normalize_text(
        reference
    )

    matches = []

    for element in elements:

        path = normalize_text(element["path"])

        if (
            path == normalized_reference
            or path.endswith(
                " > " + normalized_reference
            )
        ):
            matches.append(element)

    if len(matches) == 0:

        raise ValueError(
            f"Dispositivo não encontrado: "
            f"{reference}"
        )

    if len(matches) > 1:

        raise ValueError(
            f"Referência ambígua: {reference}"
        )

    return matches[0]


def find_relevant_chunks(device, chunks):

    device_text = normalize_text(
        device["text"]
    )

    if not device_text:

        raise ValueError(
            "Dispositivo com texto vazio."
        )

    relevant_chunks = []

    for chunk in chunks:

        chunk_text = normalize_text(
            chunk["text"]
        )

        if device_text in chunk_text:

            relevant_chunks.append(
                chunk["id"]
            )

    return relevant_chunks


def map_question(question, elements, datasets):

    mapping = {
        "id": question["id"],
        "document": question["document"],
        "question": question["question"],
        "category": question["category"],
        "expected_answer": question["expected_answer"],
        "relevant_devices": [],
        "strategies": {}
    }

    for strategy in STRATEGIES:

        mapping["strategies"][strategy] = {
            "relevant_chunks": [],
            "devices_without_full_match": []
        }

    for reference in question["relevant_devices"]:

        device = find_device(
            reference,
            elements
        )

        device_mapping = {
            "reference": reference,
            "full_path": device["path"],
            "text": device["text"],
            "chunks": {}
        }

        for strategy in STRATEGIES:

            chunks = datasets[strategy]

            relevant_chunks = find_relevant_chunks(
                device,
                chunks
            )

            device_mapping["chunks"][strategy] = (
                relevant_chunks
            )

            strategy_mapping = (
                mapping["strategies"][strategy]
            )

            strategy_mapping["relevant_chunks"].extend(
                relevant_chunks
            )

            if not relevant_chunks:

                strategy_mapping[
                    "devices_without_full_match"
                ].append(reference)

        mapping["relevant_devices"].append(
            device_mapping
        )

    # Remove identificadores duplicados.
    for strategy in STRATEGIES:

        strategy_mapping = (
            mapping["strategies"][strategy]
        )

        strategy_mapping["relevant_chunks"] = sorted(
            set(strategy_mapping["relevant_chunks"])
        )

    return mapping


def print_mapping(mapping):

    print("\n" + "=" * 60)
    print("ID:", mapping["id"])
    print("Pergunta:", mapping["question"])
    print("=" * 60)

    for device in mapping["relevant_devices"]:

        print("\nDispositivo:", device["reference"])

        for strategy in STRATEGIES:

            chunks = device["chunks"][strategy]

            print(
                f"{strategy}:",
                ", ".join(chunks)
                if chunks
                else "Sem correspondência integral"
            )

    print("\nRESUMO")

    for strategy in STRATEGIES:

        result = mapping["strategies"][strategy]

        print("\nEstratégia:", strategy)

        print(
            "Fragmentos relevantes:",
            len(result["relevant_chunks"])
        )

        print(
            "Dispositivos sem correspondência integral:",
            len(result["devices_without_full_match"])
        )


if __name__ == "__main__":

    questions = load_json(QUESTIONS_PATH)
    elements = load_json(HIERARCHY_PATH)

    datasets = {}

    for strategy in STRATEGIES:

        path = (
            PROCESSED_DIR /
            f"lgpd_{strategy}_chunks.json"
        )

        datasets[strategy] = load_json(path)

    mappings = []

    print("Perguntas carregadas:", len(questions))
    print("Elementos normativos:", len(elements))

    for question in questions:

        mapping = map_question(
            question,
            elements,
            datasets
        )

        mappings.append(mapping)

        print_mapping(mapping)

    OUTPUT_PATH.write_text(
        json.dumps(
            mappings,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    total_devices = sum(
        len(mapping["relevant_devices"])
        for mapping in mappings
    )

    print("\n" + "=" * 60)
    print("RESULTADO GERAL")
    print("=" * 60)

    print("Perguntas processadas:", len(mappings))

    print(
        "Referências normativas:",
        total_devices
    )

    for strategy in STRATEGIES:

        missing = sum(
            len(
                mapping["strategies"][strategy][
                    "devices_without_full_match"
                ]
            )
            for mapping in mappings
        )

        covered = total_devices - missing

        print("\nEstratégia:", strategy)

        print(
            "Referências integralmente mapeadas:",
            covered
        )

        print(
            "Referências sem correspondência integral:",
            missing
        )

        print(
            "Cobertura integral (%):",
            round(
                100 * covered / total_devices,
                2
            )
        )

    print("\nArquivo gerado:", OUTPUT_PATH)
