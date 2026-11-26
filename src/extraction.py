from bs4 import BeautifulSoup
from pathlib import Path
import re

def extract_text(file_path):
    with open(file_path, "rb") as file:
        soup = BeautifulSoup(file.read(), "html.parser")

    for element in soup.find_all(["script", "style"]):
        element.decompose()

    for element in soup.find_all(["strike", "s", "del"]):
        if element.parent is not None:
            element.decompose()

    for element in soup.find_all(
        style=re.compile(r"line-through", re.IGNORECASE)
    ):
        if element.parent is not None:
            element.decompose()

    for link in soup.find_all("a"):
        content = link.get_text(" ", strip=True)

        if re.match(
            r"^\(?\s*(Redação dada|Incluído pela|"
            r"Incluída pela|Vide|Vigência|"
            r"Produção de efeitos|Regulamentação)\b",
            content,
            re.IGNORECASE
        ):
            link.decompose()

    paragraphs = []

    for element in soup.find_all(
        ["p", "h1", "h2", "h3", "h4", "h5", "h6"]
    ):
        text = element.get_text(separator=" ", strip=True)

        text = re.sub(r"\s+", " ", text).strip()

        text = re.sub(
            r"^(Art\.\s*)(\d+(?:\s+\d+)+)",
            lambda m: m[1] + re.sub(r"\s+", "", m[2]),
            text
        )
        text = re.sub(r"\s+([,.;:])", r"\1", text)

        if re.match(r"^Brasília\s*,\s*\d+", text, re.IGNORECASE):
            break

        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


if __name__ == "__main__":
    input_path = Path("data/raw/lgpd_compilado.html")
    output_path = Path("data/processed/lgpd.txt")

    text = extract_text(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    print("Arquivo utilizado:", input_path.resolve())
    print("Quantidade de caracteres:", len(text))
    print("Quantidade de linhas:", len(text.splitlines()))
    print("Arquivo gerado:", output_path)
    print("\nPrévia do texto:\n")
    print(text[:2000])
