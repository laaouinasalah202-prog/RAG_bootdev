import argparse
import search
import build
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    build_parser = subparsers.add_parser("build", help="build the invert_indix")
    search_parser.add_argument("query", type=str, help="Search query")
    # build_parser.add_argument("term", type=str, help="search term")
    args = parser.parse_args()

    match args.command:
        case "search":
            search.simple_search(args.query)
        case "build":
            inver_indx = build.InvertedIndex()
            inver_indx.build()
            inver_indx.save()
            # parser.print_help()

if __name__ == "__main__":
    main()
