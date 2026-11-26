
import os
import json
import time
import hashlib

from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version

from dotenv import load_dotenv
from google import genai
from google.genai import types


ROOT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.5-flash-lite"

TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 1024
TOP_K = 5

REQUEST_INTERVAL = 5
MAX_RETRIES = 3

PROMPT_VERSION = "1.0"

INPUT_PATH = (
    ROOT_DIR / "results" / "pilot_results.json"
)

OUTPUT_PATH = (
    ROOT_DIR / "results" /
    "pilot_generated_answers.json"
)


PROMPT_TEMPLATE = """
Você é um assistente especializado na interpretação
de documentos normativos brasileiros.

Responda à pergunta utilizando exclusivamente
as informações presentes no contexto fornecido.

Regras:
- Não utilize conhecimentos externos ao contexto.
- Não invente informações ou dispositivos normativos.
- Caso o contexto seja insuficiente, informe
  explicitamente essa limitação.
- Caso o contexto permita uma resposta parcial,
  apresente as informações disponíveis e indique
  o que não foi encontrado.
- Cite os dispositivos normativos quando estiverem
  identificados no contexto.
- Responda em português, de forma objetiva e completa.

CONTEXTO:

{context}

PERGUNTA:

{question}

RESPOSTA:
""".strip()


def load_json(path):

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def build_context(results):

    if len(results) != TOP_K:

        raise ValueError(
            f"Esperados {TOP_K} fragmentos, "
            f"recebidos {len(results)}."
        )

    ordered_results = sorted(
        results,
        key=lambda item: item["rank"]
    )

    ranks = [
        result["rank"]
        for result in ordered_results
    ]

    if ranks != list(range(1, TOP_K + 1)):
        raise ValueError(
            "Posições dos fragmentos inválidas."
        )

    fragments = []

    for result in ordered_results:

        fragment = (
            f"[Fragmento {result['rank']}]\n"
            f"{result['text']}"
        )

        fragments.append(fragment)

    return "\n\n".join(fragments)


def build_prompt(question, context):

    return PROMPT_TEMPLATE.format(
        question=question,
        context=context
    )


def build_experiment_config():

    input_hash = hashlib.sha256(
        INPUT_PATH.read_bytes()
    ).hexdigest()

    prompt_hash = hashlib.sha256(
        PROMPT_TEMPLATE.encode("utf-8")
    ).hexdigest()

    return {
        "provider": "Google Gemini",
        "model": MODEL_NAME,
        "sdk_version": version("google-genai"),
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "top_k": TOP_K,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "input_sha256": input_hash
    }


def generate_answer(client, prompt):

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            start = time.perf_counter()

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS
                )
            )

            elapsed = time.perf_counter() - start

            if not response.text:

                raise RuntimeError(
                    "Resposta textual vazia."
                )

            usage = response.usage_metadata

            finish_reason = None

            if response.candidates:

                reason = (
                    response.candidates[0].finish_reason
                )

                finish_reason = (
                    reason.value if reason else None
                )

            return {
                "answer": response.text,
                "model_version": response.model_version,

                "generation_time_seconds": elapsed,

                "input_tokens": (
                    usage.prompt_token_count
                    if usage else None
                ),

                "output_tokens": (
                    usage.candidates_token_count
                    if usage else None
                ),

                "thinking_tokens": (
                    usage.thoughts_token_count
                    if usage else None
                ),

                "total_tokens": (
                    usage.total_token_count
                    if usage else None
                ),

                "finish_reason": finish_reason,
                "attempts": attempt
            }

        except Exception as error:

            status = getattr(error, "code", None)

            retryable = status in [
                429, 500, 502, 503, 504
            ]

            print(
                f"Falha na tentativa "
                f"{attempt}/{MAX_RETRIES}."
            )

            if not retryable or attempt == MAX_RETRIES:
                raise

            wait_time = 15 * (2 ** (attempt - 1))

            print(
                f"Aguardando {wait_time} segundos..."
            )

            time.sleep(wait_time)


def save_results(experiment):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_path = OUTPUT_PATH.with_suffix(".tmp")

    temporary_path.write_text(
        json.dumps(
            experiment,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    temporary_path.replace(OUTPUT_PATH)


def load_or_create_experiment(config):

    if OUTPUT_PATH.exists():

        experiment = load_json(OUTPUT_PATH)

        if experiment["configuration"] != config:

            raise ValueError(
                "A configuração experimental foi alterada. "
                "Arquive o resultado anterior antes "
                "de iniciar um novo experimento."
            )

        return experiment

    return {
        "configuration": config,
        "prompt_template": PROMPT_TEMPLATE,

        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "answers": []
    }


def print_result(result):

    print("\n" + "=" * 60)

    print("Pergunta:", result["question_id"])
    print("Estratégia:", result["strategy"])

    print("=" * 60)

    print("\n", result["question"])

    print("\nRESPOSTA GERADA:")
    print(result["answer"])

    print("\nTokens de entrada:", result["input_tokens"])

    print("Tokens de saída:", result["output_tokens"])

    print("Tokens totais:", result["total_tokens"])

    print(
        "Tempo de geração:",
        round(
            result["generation_time_seconds"],
            4
        ),
        "segundos"
    )

    print(
        "Motivo de finalização:",
        result["finish_reason"]
    )

    if result["finish_reason"] != "STOP":

        print(
            "ATENÇÃO: revisar motivo de finalização."
        )


def main():

    if not API_KEY:

        raise ValueError(
            "GEMINI_API_KEY não encontrada."
        )

    retrieval_results = load_json(INPUT_PATH)

    config = build_experiment_config()

    experiment = load_or_create_experiment(config)

    input_keys = [
        (
            result["question_id"],
            result["strategy"]
        )
        for result in retrieval_results
    ]

    if len(input_keys) != len(set(input_keys)):

        raise ValueError(
            "Resultados de recuperação duplicados."
        )

    completed = {
        (
            answer["question_id"],
            answer["strategy"]
        )
        for answer in experiment["answers"]
    }

    print("=" * 60)
    print("GERAÇÃO DE RESPOSTAS RAG")
    print("=" * 60)

    print("Modelo:", MODEL_NAME)
    print("Respostas esperadas:", len(retrieval_results))
    print("Respostas já geradas:", len(completed))

    client = genai.Client(api_key=API_KEY)

    try:

        for retrieval in retrieval_results:

            question_id = retrieval["question_id"]
            strategy = retrieval["strategy"]

            key = (question_id, strategy)

            if key in completed:

                print(
                    f"Ignorando {question_id} - "
                    f"{strategy}: já processado."
                )

                continue

            context = build_context(
                retrieval["results"]
            )

            prompt = build_prompt(
                retrieval["question"],
                context
            )

            generated = generate_answer(
                client,
                prompt
            )

            result = {
                "question_id": question_id,
                "question": retrieval["question"],
                "category": retrieval["category"],
                "strategy": strategy,

                "retrieved_chunks": (
                    retrieval["retrieved_chunks"]
                ),

                "retrieved_tokens_embedding": (
                    retrieval["retrieved_tokens"]
                ),

                "context": context,
                "prompt": prompt,

                "generated_at": datetime.now(
                    timezone.utc
                ).isoformat(),

                **generated
            }

            experiment["answers"].append(result)

            save_results(experiment)

            completed.add(key)

            print_result(result)

            print(
                "\nProgresso:",
                len(completed),
                "/",
                len(retrieval_results)
            )

            if len(completed) < len(retrieval_results):

                time.sleep(REQUEST_INTERVAL)

    finally:

        client.close()

    print("\n" + "=" * 60)
    print("GERAÇÃO CONCLUÍDA")
    print("=" * 60)

    print(
        "Total de respostas:",
        len(experiment["answers"])
    )

    print("Arquivo gerado:", OUTPUT_PATH)


if __name__ == "__main__":

    main()
