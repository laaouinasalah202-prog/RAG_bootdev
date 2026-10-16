import argparse
import json
from lib.hybrid_search import HybridSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    
    with open("data/golden_dataset.json", "r") as f:
        golden_data = json.load(f)
    k = limit
    hybrid_search = HybridSearch()
    doc_map = hybrid_search.idx.docmap
    retrive_map = []
    for eval in (golden_data["test_cases"]):
        rrf = hybrid_search.rrf_search(eval["query"], 60, limit)
        retrieved_titles = [doc_map[id]["title"] for id, _ in rrf][:k]
        relevant_id = [rel for rel in retrieved_titles if rel in eval["relevant_docs"]]
        precision = len(relevant_id)/k
        retrive_map.append(
            {"precision": precision, "retrieved": retrieved_titles,
                "query": eval["query"], "relevant": eval["relevant_docs"]
            }
        )
    for res in retrive_map:
        print(f'k = {k}\n'
            f'- Query: {res["query"]}\n'
            f'- Precision@{k}: {round(res["precision"], 4)}\n'
            f'- Retrieved: {res["retrieved"]}\n'
            f'- Relevant: {res["relevant"]}'
            )

if __name__ == "__main__":
    main()
