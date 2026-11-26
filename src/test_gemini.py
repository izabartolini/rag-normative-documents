
import os
import time

from pathlib import Path
from importlib.metadata import version

from dotenv import load_dotenv
from google import genai
from google.genai import types


ROOT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.5-flash-lite"


if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY não encontrada no arquivo .env"
    )


def test_connection():

    client = genai.Client(api_key=API_KEY)

    prompt = """
    Explique brevemente, em português,
    o que é Retrieval-Augmented Generation (RAG).
    """

    start = time.perf_counter()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=1024
        )
    )

    elapsed = time.perf_counter() - start

    if not response.text:
        raise RuntimeError(
            "O modelo não retornou uma resposta textual."
        )

    print("=" * 60)
    print("TESTE DE CONEXÃO COM O GEMINI")
    print("=" * 60)

    print("Modelo solicitado:", MODEL_NAME)

    print(
        "Versão do modelo:",
        response.model_version
    )

    print(
        "Versão do SDK:",
        version("google-genai")
    )

    print("\nResposta:")
    print(response.text)

    print(
        "\nTempo da requisição:",
        round(elapsed, 4),
        "segundos"
    )

    usage = response.usage_metadata

    if usage:

        print("\nCONSUMO DE TOKENS")

        print(
            "Tokens de entrada:",
            usage.prompt_token_count
        )

        print(
            "Tokens de saída:",
            usage.candidates_token_count
        )

        print(
            "Tokens de pensamento:",
            usage.thoughts_token_count
        )

        print(
            "Total de tokens:",
            usage.total_token_count
        )

    if response.candidates:
        print(
            "Motivo de finalização:",
            response.candidates[0].finish_reason
        )

    client.close()


if __name__ == "__main__":

    test_connection()
