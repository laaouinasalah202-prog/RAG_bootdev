import argparse
import search
import build

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    tf_parser = subparsers.add_parser("tf", help="find term frequencies")
    idf_parser = subparsers.add_parser("idf", help="Calculate the IDF for the term")
    tfidf_parser = subparsers.add_parser("tfidf", help="calculate TF-IDF")
    build_parser = subparsers.add_parser("build", help="build the invert_indix")
    tfidf_parser.add_argument("id", type=int, help="the id of the doc")
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

if __name__ == "__main__":
    main()
