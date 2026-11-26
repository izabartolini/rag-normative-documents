
import csv
import json

from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]

EVALUATION_DIR = ROOT / "evaluation"
RESULTS_DIR = ROOT / "results"

SCORES_PATH = EVALUATION_DIR / "textual_scores.csv"

ANSWERS_PATH = RESULTS_DIR / "pilot_generated_answers.json"

RETRIEVAL_PATH = RESULTS_DIR / "pilot_results.json"

OUTPUT_CSV = RESULTS_DIR / "textual_metrics.csv"

OUTPUT_JSON = RESULTS_DIR / "textual_summary.json"

STRATEGIES = [
    "fixed",
    "hierarchical"
]

CRITERIA = [
    "factual_correctness",
    "completeness",
    "faithfulness"
]

def validate_files():

    required_files = [
        SCORES_PATH,
        ANSWERS_PATH,
        RETRIEVAL_PATH
    ]

    missing_files = [
        path
        for path in required_files
        if not path.is_file()
    ]

    if missing_files:

        print("\nArquivos não encontrados:")

        for path in missing_files:
            print("-", path)

        raise FileNotFoundError(
            "Um ou mais arquivos obrigatórios não existem."
        )

    print("\nArquivos de entrada encontrados:")

    for path in required_files:
        print("[OK]", path.name)

def load_json(path):

    try:

        return json.loads(
            path.read_text(encoding="utf-8-sig")
        )

    except json.JSONDecodeError as error:

        raise ValueError(
            f"Arquivo JSON inválido: {path.name}. "
            f"Linha: {error.lineno}."
        ) from error


def get_key(item):

    question_id = str(
        item["question_id"]
    ).strip()

    strategy = str(
        item["strategy"]
    ).strip().lower()

    if not question_id:

        raise ValueError(
            "Identificador de pergunta vazio."
        )

    if strategy not in STRATEGIES:

        raise ValueError(
            f"Estratégia inválida: {strategy!r}"
        )

    return question_id, strategy


def index_records(records, source):

    indexed = {}

    for record in records:

        key = get_key(record)

        if key in indexed:

            raise ValueError(
                f"Registro duplicado em {source}: {key}"
            )

        indexed[key] = record

    return indexed


def load_scores():

    scores = []

    required_columns = {
        "question_id",
        "strategy",
        *CRITERIA
    }

    with SCORES_PATH.open(
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        available_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns - available_columns
        )

        if missing_columns:

            raise ValueError(
                "Colunas ausentes em textual_scores.csv: "
                f"{sorted(missing_columns)}"
            )

        for line_number, row in enumerate(
            reader,
            start=2
        ):

            if not any(
                str(value or "").strip()
                for value in row.values()
            ):
                continue

            question_id, strategy = get_key(row)

            row["question_id"] = question_id
            row["strategy"] = strategy

            for criterion in CRITERIA:

                try:

                    value = int(row[criterion])

                except (ValueError, TypeError) as error:

                    raise ValueError(
                        f"Nota inválida na linha {line_number}. "
                        f"Critério: {criterion}. "
                        f"Valor: {row[criterion]!r}"
                    ) from error

                if value not in [0, 1, 2]:

                    raise ValueError(
                        f"Nota fora do intervalo permitido "
                        f"na linha {line_number}: "
                        f"{criterion} = {value}"
                    )

                row[criterion] = value

            row["total_score"] = sum(
                row[criterion]
                for criterion in CRITERIA
            )

            scores.append(row)

    if not scores:

        raise ValueError(
            "O arquivo textual_scores.csv está vazio."
        )

    return scores


def load_answers():

    experiment = load_json(ANSWERS_PATH)

    if not isinstance(experiment, dict):

        raise ValueError(
            "Estrutura inválida em "
            "pilot_generated_answers.json."
        )

    if "answers" not in experiment:

        raise ValueError(
            "A chave 'answers' não foi encontrada."
        )

    answers = experiment["answers"]

    if not isinstance(answers, list) or not answers:

        raise ValueError(
            "Nenhuma resposta gerada foi encontrada."
        )

    return answers


def load_retrieval():

    retrieval = load_json(RETRIEVAL_PATH)

    if not isinstance(retrieval, list) or not retrieval:

        raise ValueError(
            "Estrutura inválida ou vazia em "
            "pilot_results.json."
        )

    return retrieval

def validate_record_sets(
    score_index,
    answer_index,
    retrieval_index
):

    score_keys = set(score_index)

    answer_keys = set(answer_index)

    retrieval_keys = set(retrieval_index)

    print("\n" + "=" * 60)
    print("VALIDAÇÃO DOS REGISTROS")
    print("=" * 60)

    print("Pontuações:", len(score_keys))

    print("Respostas geradas:", len(answer_keys))

    print("Resultados da recuperação:", len(retrieval_keys))

    if score_keys == answer_keys == retrieval_keys:

        print("\n[OK] Os conjuntos de registros correspondem.")

        return

    all_keys = (
        score_keys
        | answer_keys
        | retrieval_keys
    )

    sources = {
        "textual_scores.csv": score_keys,

        "pilot_generated_answers.json": answer_keys,

        "pilot_results.json": retrieval_keys
    }

    print("\n" + "=" * 60)
    print("DIAGNÓSTICO DE INCOMPATIBILIDADE")
    print("=" * 60)

    for source, keys in sources.items():

        missing = sorted(all_keys - keys)

        print(f"\nArquivo: {source}")

        print("Quantidade de registros:", len(keys))

        print("Registros ausentes:")

        if missing:

            for question_id, strategy in missing:

                print(
                    f"  - {question_id} | {strategy}"
                )

        else:

            print("  Nenhum.")

        print("\nRegistros encontrados:")

        for question_id, strategy in sorted(keys):

            print(
                f"  - {question_id} | {strategy}"
            )

    raise ValueError(
        "Os conjuntos de resultados não correspondem. "
        "Verifique o diagnóstico apresentado acima."
    )

def validate_pairs(keys):

    question_ids = {
        question_id
        for question_id, strategy in keys
    }

    for question_id in sorted(question_ids):

        for strategy in STRATEGIES:

            key = (question_id, strategy)

            if key not in keys:

                raise ValueError(
                    f"Par experimental incompleto: {key}"
                )

    print(
        "[OK] Todas as perguntas possuem "
        "resultados nas duas estratégias."
    )

def combine_results(scores, answers, retrieval):

    score_index = index_records(
        scores,
        "textual_scores.csv"
    )

    answer_index = index_records(
        answers,
        "pilot_generated_answers.json"
    )

    retrieval_index = index_records(
        retrieval,
        "pilot_results.json"
    )

    validate_record_sets(
        score_index,
        answer_index,
        retrieval_index
    )

    validate_pairs(
        set(score_index)
    )

    combined = []

    for key, score in score_index.items():

        answer = answer_index[key]

        retrieved = retrieval_index[key]

        question_id, strategy = key

        if answer.get("finish_reason") != "STOP":

            raise ValueError(
                f"Resposta não finalizada normalmente: {key}. "
                f"Motivo: {answer.get('finish_reason')}"
            )

        if not str(answer.get("answer") or "").strip():

            raise ValueError(
                f"Resposta textual vazia: {key}"
            )

        if answer["question"] != retrieved["question"]:

            raise ValueError(
                f"Perguntas incompatíveis: {key}"
            )

        if (
            answer["retrieved_chunks"]
            != retrieved["retrieved_chunks"]
        ):

            raise ValueError(
                f"Fragmentos incompatíveis: {key}"
            )

        ordered_fragments = sorted(
            retrieved["results"],
            key=lambda fragment: fragment["rank"]
        )

        expected_context = "\n\n".join(
            f"[Fragmento {fragment['rank']}]\n"
            f"{fragment['text']}"
            for fragment in ordered_fragments
        )

        if answer["context"] != expected_context:

            raise ValueError(
                f"Conteúdo do contexto incompatível: {key}"
            )

        generation_fields = [
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "generation_time_seconds"
        ]

        for field in generation_fields:

            value = answer.get(field)

            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value < 0
            ):

                raise ValueError(
                    f"Métrica de geração inválida: "
                    f"{key} | {field} = {value}"
                )

        combined.append({

            "question_id": question_id,

            "question": answer["question"],

            "category": answer["category"],

            "strategy": strategy,

            "factual_correctness":
                score["factual_correctness"],

            "completeness":
                score["completeness"],

            "faithfulness":
                score["faithfulness"],

            "total_score":
                score["total_score"],

            "precision_at_5":
                retrieved["precision_at_5"],

            "recall_at_5":
                retrieved["recall_at_5"],

            "reciprocal_rank_at_5":
                retrieved["reciprocal_rank_at_5"],

            "hit_at_5":
                retrieved["hit_at_5"],

            "device_recall_at_5":
                retrieved["device_recall_at_5"],

            "retrieved_tokens_embedding":
                retrieved["retrieved_tokens"],

            "retrieval_time_ms":
                retrieved["retrieval_time_ms"],

            "input_tokens":
                answer["input_tokens"],

            "output_tokens":
                answer["output_tokens"],

            "thinking_tokens":
                answer.get("thinking_tokens"),

            "total_tokens":
                answer["total_tokens"],

            "generation_time_seconds":
                answer["generation_time_seconds"],

            "model_version":
                answer.get("model_version"),

            "finish_reason":
                answer["finish_reason"],

            "attempts":
                answer.get("attempts")
        })

    print(
        "[OK] Respostas e contextos correspondem "
        "aos resultados da recuperação."
    )

    return combined

def summarize(results):

    summary = {}

    for strategy in STRATEGIES:

        rows = [
            row
            for row in results
            if row["strategy"] == strategy
        ]

        if not rows:

            raise ValueError(
                f"Estratégia sem resultados: {strategy}"
            )

        total_answers = len(rows)

        complete_answers = sum(
            row["completeness"] == 2
            for row in rows
        )

        partial_answers = sum(
            row["completeness"] == 1
            for row in rows
        )

        insufficient_answers = sum(
            row["completeness"] == 0
            for row in rows
        )

        total_score = sum(
            row["total_score"]
            for row in rows
        )

        max_possible_score = total_answers * 6

        summary[strategy] = {

            "total_answers": total_answers,

            "mean_factual_correctness": mean(
                row["factual_correctness"]
                for row in rows
            ),

            "mean_completeness": mean(
                row["completeness"]
                for row in rows
            ),

            "mean_faithfulness": mean(
                row["faithfulness"]
                for row in rows
            ),

            "total_score": total_score,

            "max_possible_score": max_possible_score,

            "mean_total_score": mean(
                row["total_score"]
                for row in rows
            ),

            "normalized_score":
                total_score / max_possible_score,

            "complete_answers": complete_answers,

            "partial_answers": partial_answers,

            "insufficient_answers": insufficient_answers,

            "complete_answer_rate":
                complete_answers / total_answers,

            "partial_answer_rate":
                partial_answers / total_answers,

            "insufficient_answer_rate":
                insufficient_answers / total_answers,

            "mean_precision_at_5": mean(
                row["precision_at_5"]
                for row in rows
            ),

            "mean_recall_at_5": mean(
                row["recall_at_5"]
                for row in rows
            ),

            "mrr_at_5": mean(
                row["reciprocal_rank_at_5"]
                for row in rows
            ),

            "hit_rate_at_5": mean(
                row["hit_at_5"]
                for row in rows
            ),

            "mean_device_recall_at_5": mean(
                row["device_recall_at_5"]
                for row in rows
            ),

            "mean_retrieved_tokens_embedding": mean(
                row["retrieved_tokens_embedding"]
                for row in rows
            ),

            "mean_retrieval_time_ms": mean(
                row["retrieval_time_ms"]
                for row in rows
            ),

            "total_input_tokens": sum(
                row["input_tokens"]
                for row in rows
            ),

            "total_output_tokens": sum(
                row["output_tokens"]
                for row in rows
            ),

            "total_tokens": sum(
                row["total_tokens"]
                for row in rows
            ),

            "mean_input_tokens": mean(
                row["input_tokens"]
                for row in rows
            ),

            "mean_output_tokens": mean(
                row["output_tokens"]
                for row in rows
            ),

            "mean_total_tokens": mean(
                row["total_tokens"]
                for row in rows
            ),

            # Desempenho da geração

            "total_generation_time_seconds": sum(
                row["generation_time_seconds"]
                for row in rows
            ),

            "mean_generation_time_seconds": mean(
                row["generation_time_seconds"]
                for row in rows
            )
        }

    return summary

def compare_pairs(results):

    indexed = index_records(
        results,
        "resultados consolidados"
    )

    question_ids = sorted({

        row["question_id"]

        for row in results
    })

    comparison = {

        "total_questions": len(question_ids),

        "fixed_wins": 0,

        "hierarchical_wins": 0,

        "ties": 0,

        "details": []
    }

    for question_id in question_ids:

        fixed = indexed[
            (question_id, "fixed")
        ]["total_score"]

        hierarchical = indexed[
            (question_id, "hierarchical")
        ]["total_score"]

        if fixed > hierarchical:

            winner = "fixed"

            comparison["fixed_wins"] += 1

        elif hierarchical > fixed:

            winner = "hierarchical"

            comparison["hierarchical_wins"] += 1

        else:

            winner = "tie"

            comparison["ties"] += 1

        comparison["details"].append({

            "question_id": question_id,

            "fixed_score": fixed,

            "hierarchical_score": hierarchical,

            "difference": hierarchical - fixed,

            "winner": winner
        })

    return comparison

def percentage_change(baseline, alternative):

    if baseline == 0:

        return None

    return (
        (alternative - baseline)
        / baseline
    ) * 100


def compare_strategies(summary):

    fixed = summary["fixed"]

    hierarchical = summary["hierarchical"]

    metrics = [

        "mean_factual_correctness",

        "mean_completeness",

        "mean_faithfulness",

        "mean_total_score",

        "mean_precision_at_5",

        "mean_recall_at_5",

        "mrr_at_5",

        "hit_rate_at_5",

        "mean_device_recall_at_5",

        "mean_retrieved_tokens_embedding",

        "total_input_tokens",

        "total_output_tokens",

        "total_tokens",

        "mean_generation_time_seconds"
    ]

    comparison = {}

    for metric in metrics:

        fixed_value = fixed[metric]

        hierarchical_value = hierarchical[metric]

        change = percentage_change(
            fixed_value,
            hierarchical_value
        )

        comparison[metric] = {

            "fixed": fixed_value,

            "hierarchical": hierarchical_value,

            "absolute_difference":
                hierarchical_value - fixed_value,

            "percentage_change":
                change
        }

    comparison["complete_answer_rate"] = {

        "fixed": fixed["complete_answer_rate"],

        "hierarchical":
            hierarchical["complete_answer_rate"],

        "difference_percentage_points": (

            hierarchical["complete_answer_rate"]

            - fixed["complete_answer_rate"]

        ) * 100
    }

    return comparison

def save_results(results, summary):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not results:

        raise ValueError(
            "Não existem resultados para salvar."
        )

    with OUTPUT_CSV.open(

        "w",

        encoding="utf-8-sig",

        newline=""

    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=list(results[0].keys()),

            delimiter=";"
        )

        writer.writeheader()

        writer.writerows(results)

    OUTPUT_JSON.write_text(

        json.dumps(

            summary,

            ensure_ascii=False,

            indent=4
        ),

        encoding="utf-8"
    )

def print_summary(summary):

    print("\n" + "=" * 60)

    print("RESULTADOS DA AVALIAÇÃO TEXTUAL")

    print("=" * 60)

    for strategy, metrics in summary["strategies"].items():

        print("\n" + "-" * 60)

        print("ESTRATÉGIA:", strategy.upper())

        print("-" * 60)

        for metric, value in metrics.items():

            if isinstance(value, float):

                print(f"{metric}: {value:.6f}")

            else:

                print(f"{metric}: {value}")

    paired = summary["paired_comparison"]

    print("\n" + "=" * 60)

    print("COMPARAÇÃO PAREADA")

    print("=" * 60)

    print(
        "Vitórias da estratégia fixa:",
        paired["fixed_wins"]
    )

    print(
        "Vitórias da estratégia hierárquica:",
        paired["hierarchical_wins"]
    )

    print(
        "Empates:",
        paired["ties"]
    )

    print("\nResultados por pergunta:")

    for detail in paired["details"]:

        print(

            f"{detail['question_id']} | "

            f"Fixa: {detail['fixed_score']} | "

            f"Hierárquica: {detail['hierarchical_score']} | "

            f"Resultado: {detail['winner']}"
        )

    print("\n" + "=" * 60)

    print("VARIAÇÕES PERCENTUAIS")

    print("=" * 60)

    for metric, values in (
        summary["strategy_comparison"].items()
    ):

        change = values.get("percentage_change")

        if change is not None:

            print(f"{metric}: {change:+.2f}%")

        elif "difference_percentage_points" in values:

            difference = values[
                "difference_percentage_points"
            ]

            print(f"{metric}: {difference:+.2f} p.p.")

        else:

            print(f"{metric}: não calculável")

def main():

    print("=" * 60)

    print("AVALIAÇÃO TEXTUAL DO SISTEMA RAG")

    print("=" * 60)

    validate_files()

    scores = load_scores()

    answers = load_answers()

    retrieval = load_retrieval()

    results = combine_results(
        scores,
        answers,
        retrieval
    )

    strategy_summary = summarize(results)

    paired_comparison = compare_pairs(results)

    strategy_comparison = compare_strategies(
        strategy_summary
    )

    summary = {

        "evaluation_method":
            "AI-assisted rubric, pending human review",

        "total_evaluated_answers": len(results),

        "scoring_configuration": {

            "criteria": CRITERIA,

            "minimum_score_per_criterion": 0,

            "maximum_score_per_criterion": 2,

            "maximum_score_per_answer": 6
        },

        "strategies": strategy_summary,

        "paired_comparison": paired_comparison,

        "strategy_comparison": strategy_comparison
    }

    save_results(
        results,
        summary
    )

    print_summary(summary)

    print("\n" + "=" * 60)

    print("AVALIAÇÃO CONCLUÍDA COM SUCESSO")

    print("=" * 60)

    print("\nTotal de respostas avaliadas:", len(results))

    print("\nArquivos gerados:")

    print(OUTPUT_CSV)

    print(OUTPUT_JSON)


if __name__ == "__main__":

    main()
