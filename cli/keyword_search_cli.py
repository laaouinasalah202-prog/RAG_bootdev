import argparse
from operator import sub
import search
from lib import build

def main() -> None:
    BM25_K1 = 1.5
    BM25_B = 0.75
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    tf_parser = subparsers.add_parser("tf", help="find term frequencies")
    idf_parser = subparsers.add_parser("idf", help="Calculate the IDF for the term")
    tfidf_parser = subparsers.add_parser("tfidf", help="calculate TF-IDF")
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    build_parser = subparsers.add_parser("build", help="build the invert_indix")
    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    tfidf_parser.add_argument("id", type=int, help="the id of the doc")
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", default=5,type=int, help="number of top result")
    tfidf_parser.add_argument("term", type=str, help="the term to calculate tf-idf")
    idf_parser.add_argument("term", type=str, help="term to calculate IDF")
    search_parser.add_argument("query", type=str, help="Search query")
    tf_parser.add_argument("ID", type=int, help="the id of the doc")
    tf_parser.add_argument("term", type=str, help="the term to calculate frequencies")
    args = parser.parse_args()

    invert_index = build.InvertedIndex()
    match args.command:
        case "search":
            search.simple_search(args.query, invert_index)
            
        case "build":
            invert_index.build()
            invert_index.save()
            
        case "tf":
            invert_index.load()
            tr_fr = invert_index.get_tf(args.ID, args.term)
            print(f"the term {args.term} appears {tr_fr} times in the document id {args.ID}")

        case "idf":
            invert_index.load()
            idf = invert_index.calculate_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            invert_index.load()
            tf_idf = invert_index.calculate_tfidf(args.term, args.id)
            print(f"TF-IDF score of '{args.term}' in document '{args.id}': {tf_idf:.2f}")
        case "bm25idf":
            invert_index.load()
            bm25idf = invert_index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf": 
            invert_index.load()
            bm25tf = invert_index.get_bm25_tf(args.doc_id, args.term)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            invert_index.load()
            result = invert_index.bm25_search(args.query, args.limit)
            
            for id, score in (result):
                print(f'({1}) {invert_index.docmap[id]["title"]} - Score: {score:.2f}')


if __name__ == "__main__":
    main()
