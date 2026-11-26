
from sentence_transformers import SentenceTransformer

from retrieval import (
    MODEL_NAME,
    load_strategy,
    encode_query,
    retrieve
)

QUESTION = "O que são dados pessoais sensíveis?"

EXPECTED_TEXT = (
    "dado pessoal sensível: dado pessoal sobre"
)


if __name__ == "__main__":

    model = SentenceTransformer(MODEL_NAME)

    query_embedding, _ = encode_query(
        model,
        QUESTION
    )

    print("Pergunta:", QUESTION)

    for strategy in ["fixed", "hierarchical"]:

        chunks, embeddings = load_strategy(
            strategy
        )

        results, _ = retrieve(
            query_embedding,
            chunks,
            embeddings,
            top_k=len(chunks)
        )

        print("\n" + "=" * 60)
        print("ESTRATÉGIA:", strategy.upper())
        print("=" * 60)

        relevant_results = []

        for result in results:

            normalized_text = " ".join(
                result["text"].casefold().split()
            )

            if EXPECTED_TEXT.casefold() in normalized_text:

                relevant_results.append(result)

                print("\nFragmento esperado encontrado!")
                print("ID:", result["id"])
                print("Posição:", result["rank"])

                print(
                    "Similaridade:",
                    round(result["similarity"], 6)
                )

                print("Tokens:", result["token_count"])
                print("Conteúdo:", result["text"])

        if not relevant_results:
            print(
                "Nenhum fragmento contém "
                "o trecho esperado."
            )

        else:

            first_rank = relevant_results[0]["rank"]

            print("\nRESUMO")
            print("Total de fragmentos:", len(chunks))

            print(
                "Fragmentos com o trecho esperado:",
                len(relevant_results)
            )

            print(
                "Primeira posição relevante:",
                first_rank
            )

            print("Hit@5:", int(first_rank <= 5))
            print("Hit@10:", int(first_rank <= 10))
            print("Hit@20:", int(first_rank <= 20))

            print(
                "Reciprocal Rank completo:",
                round(1 / first_rank, 6)
            )
