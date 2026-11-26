from pathlib import Path
import re

text = Path("data/processed/lgpd.txt").read_text(
    encoding="utf-8"
)

patterns = {
    "Capítulos": r"(?m)^\s*CAPÍTULO\s+[IVXLCDM]+\b",
    "Seções": r"(?mi)^\s*Seção\s+[IVXLCDM]+\b",
    "Artigos": r"(?m)^\s*Art\.\s*\d+",
    "Parágrafos numerados": r"(?m)^\s*§\s*\d+",
    "Parágrafos únicos": r"(?mi)^\s*Parágrafo único",
    "Incisos": r"(?m)^\s*[IVXLCDM]+\s*[-–—]",
    "Alíneas": r"(?m)^\s*[a-z]\)"
}

for name, pattern in patterns.items():
    matches = re.findall(pattern, text)

    print(f"{name}: {len(matches)}")
    print(f"Exemplos: {matches[:5]}")
    print()
