"""Create answer.csv using the selected local hybrid retrieval ensemble."""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from avito_retrieval import (
    FridaEmbedder,
    LexicalRetriever,
    chunk_articles,
    load_dataset,
    load_or_create_embeddings,
    rank_article_scores,
    reciprocal_rank_fusion,
    score_query_to_article,
    score_query_to_query,
)


DATA_PATH = "candidate_public/candidate_data"
CACHE_PATH = "cache"
FUSION_WEIGHTS = (2.0, 1.5, 3.0, 0.5)


def encode_bge(texts: list[str], cache_name: str) -> np.ndarray:
    """Encode BGE-M3 locally and persist vectors for repeatable submission runs."""
    model = SentenceTransformer(
        "models/bge-m3",
        device="mps" if torch.backends.mps.is_available() else "cpu",
        local_files_only=True,
        model_kwargs={"torch_dtype": torch.float16 if torch.backends.mps.is_available(
        ) else torch.float32},
    )
    embeddings = load_or_create_embeddings(
        f"{CACHE_PATH}/{cache_name}",
        texts,
        lambda values: model.encode(
            values,
            batch_size=8,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ),
    )
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return embeddings


def main() -> None:
    """Generate a ranked top-10 article list for every test query."""
    data = load_dataset(DATA_PATH)
    article_ids = data.articles["article_id"].to_numpy(dtype=np.int64)
    calibration_queries = data.calibration["clean_query"].tolist()
    test_queries = data.test["clean_query"].tolist()

    bge_calibration = encode_bge(calibration_queries, "bge_m3_calibration.npz")
    bge_test = encode_bge(test_queries, "bge_m3_test.npz")
    chunks = chunk_articles(data.articles)
    bge_chunks = encode_bge(chunks.texts, "bge_m3_article_chunks_384_64.npz")

    bge_query_ranking = rank_article_scores(
        score_query_to_query(
            bge_test,
            bge_calibration,
            data.calibration["ground_truth_ids"].tolist(),
            article_ids,
        ),
        article_ids,
        top_k=79,
    )
    article_ranking = rank_article_scores(
        score_query_to_article(bge_test, bge_chunks,
                               chunks.article_ids, article_ids),
        article_ids,
        top_k=100,
    )

    frida = FridaEmbedder(model_path="models/frida", batch_size=8)
    frida_calibration = load_or_create_embeddings(
        f"{CACHE_PATH}/frida_calibration_paraphrase.npz",
        calibration_queries,
        lambda values: frida.encode(values, prompt_name="paraphrase"),
    )
    frida_test = load_or_create_embeddings(
        f"{CACHE_PATH}/frida_test_paraphrase.npz",
        test_queries,
        lambda values: frida.encode(values, prompt_name="paraphrase"),
    )
    frida_query_ranking = rank_article_scores(
        score_query_to_query(
            frida_test,
            frida_calibration,
            data.calibration["ground_truth_ids"].tolist(),
            article_ids,
        ),
        article_ids,
        top_k=79,
    )

    lexical = LexicalRetriever().fit(data.calibration, data.articles)
    lexical_ranking = np.asarray(
        [lexical.rank(query, top_k=100) for query in test_queries], dtype=np.int64
    )
    fused = reciprocal_rank_fusion(
        [bge_query_ranking, frida_query_ranking, lexical_ranking, article_ranking],
        article_ids,
        weights=FUSION_WEIGHTS,
        rank_constant=100,
    )

    exact_matches = dict(
        zip(data.calibration["clean_query"],
            data.calibration["ground_truth_ids"], strict=True)
    )
    answers = []
    for query, ranked_ids in zip(test_queries, fused, strict=True):
        known_ids = exact_matches.get(query, ())
        merged_ids = list(dict.fromkeys((*known_ids, *ranked_ids)))[:10]
        answers.append(" ".join(map(str, merged_ids)))

    pd.DataFrame({"query_id": data.test["query_id"], "answer": answers}).to_csv(
        "answer.csv", index=False
    )


if __name__ == "__main__":
    main()
