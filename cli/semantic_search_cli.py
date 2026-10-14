import argparse
from lib.semantic_search import SemanticSearch, embed_text, verify_model, verify_embeddings, embed_query_text
import json

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")
    chunk_parser = subparsers.add_parser("chunk")
    chunk_parser.add_argument("text", help="text to split")
    chunk_parser.add_argument("--chunk-size", type=int, default=200)
    chunk_parser.add_argument("--overlap", type=int, help="how much to include in order to preserve the context")
    embed_text_parser = subparsers.add_parser("embed_text")
    embed_query_parser = subparsers.add_parser("embed_query", help="embed the query using the embedding model")
    search_parser = subparsers.add_parser("search", help="search for similarity of a query")
    search_parser.add_argument("--limit", default=5, help="retrieve only the top limit of vectors")
    search_parser.add_argument("query", help="the query to consider during search")
    embed_query_parser.add_argument("query", help="the query to embed")
    embed_text_parser.add_argument("text", type=str, help="text to embed")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="verify the embeddings")
    verify_parser = subparsers.add_parser("verify", help="Search movies using keywords")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            semantic_search = SemanticSearch()
            with open("data/movies.json", "r") as f:
                movies = json.load(f)
            semantic_search.load_or_create_embeddings(movies)
            resutl = semantic_search.search(args.query, args.limit)
            for n, i in enumerate(resutl):
                print(f'{n}. {i["title"]} (score: {i["cosine"]})\n')

        case "chunk":
            words = args.text.split()
            size = args.chunk_size
            overlap = args.overlap
        
            if size <= 0:
                raise SystemExit("chunk_size must be positive")
            if overlap < 0 or overlap >= size:
                raise SystemExit("overlap must be >= 0 and less than chunk_size")
        
            step = size - overlap
            chunks = [
                " ".join(words[i:i + size])
                for i in range(0, len(words), step)
            ]
        
            print(f"Chunking {len(words)} words into {len(chunks)} chunks")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
                
            
if __name__ == "__main__":
    main()
