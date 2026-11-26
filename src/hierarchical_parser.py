import re
import json
from pathlib import Path

PATTERNS = {
    "capitulo": r"^CAPÍTULO\s+[IVXLCDM]+\b",
    "secao": r"^Seção\s+[IVXLCDM]+\b",
    "artigo": r"^Art\.\s*\d+(?:º|°)?(?:-[A-Z])?",
    "paragrafo": r"^(?:§\s*\d+(?:º|°)?|Parágrafo único)",
    "inciso": r"^([IVXLCDM]+(?:-[A-Z]+)?)\s*[-–—](?=\s|$)",
    "alinea": r"^[a-z]\)"
}

LEVELS = [
    "capitulo",
    "secao",
    "artigo",
    "paragrafo",
    "inciso",
    "alinea"
]


def parse_document(text, document_name):
    hierarchy = dict.fromkeys(LEVELS)
    elements = []
    current_element = None

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        identified = False

        for level, pattern in PATTERNS.items():
            match = re.match(
                pattern, line, re.IGNORECASE
            )

            if not match:
                continue

            if level in ("capitulo", "secao"):
                hierarchy[level] = line
            elif level == "inciso":
                hierarchy[level] = "Inciso " + match.group(1).upper()
            else:
                hierarchy[level] = match.group().strip()

            position = LEVELS.index(level)

            for lower_level in LEVELS[position + 1:]:
                hierarchy[lower_level] = None

            path = [document_name]

            for item in LEVELS:
                if hierarchy[item]:
                    path.append(hierarchy[item])

            current_element = {
                "type": level,
                "path": " > ".join(path),
                "text": line
            }

            elements.append(current_element)

            identified = True
            break

        if not identified and current_element:
            current_element["text"] += " " + line

    return elements


if __name__ == "__main__":
    text = Path("data/processed/lgpd.txt").read_text(
        encoding="utf-8"
    )

    elements = parse_document(
        text,
        "Lei nº 13.709/2018 - LGPD"
    )

    output = Path("data/processed/lgpd_hierarchy.json")

    output.write_text(
        json.dumps(
            elements,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    print("Elementos processados:", len(elements))

    for element in elements[:15]:
        print("\nTipo:", element["type"])
        print("Caminho:", element["path"])
        print("Conteúdo:", element["text"])
        print("-" * 60)
