import argparse
from lib.hybrid_search import HybridSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparser.add_parser("normalize")
    weighted_search_parser = subparser.add_parser("weighted-search")
    weighted_search_parser.add_argument("query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5)
    weighted_search_parser.add_argument("--limit", type=int, default=5)
    normalize_parser.add_argument("list", nargs="*", type=float)
    
    args = parser.parse_args()
    match args.command:
        case "normalize":
            scores = args.list
            if not scores:
                pass
            elif max(scores) == min(scores):
                normalize_scores = [1.0] * len(scores)
                print(normalize_scores)
            else:
                normalize_score = [
                    round((s - min(scores)) / (max(scores) - min(scores)), 4)
                    for s in scores]
                print(normalize_score)
        case "weighted-search":
            hybrid_search = HybridSearch()
            
            best_result = hybrid_search.weighted_search(args.query, args.limit, args.alpha)
            for n, rs in enumerate(best_result, 1):
                print(f'{n}. {hybrid_search.idx.docmap[rs[0]]["title"]}\n'
                    f'Hybrid Score: {rs[1]["hybrid_score"]}\n'
                    f'BM25: {rs[1]["bm25"]}, Semantic: {rs[1]["semantic_score"]}\n'
                    f'{hybrid_search.idx.docmap[rs[0]]["description"][:100]}'
                )
if __name__ == "__main__":
    main()
