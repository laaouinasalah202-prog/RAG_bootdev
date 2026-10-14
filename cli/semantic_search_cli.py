import argparse
from lib.semantic_search import SemanticSearch, embed_text, verify_model, verify_embeddings, embed_query_text
import json
import re
def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")
    chunk_parser = subparsers.add_parser("chunk")
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk")
    semantic_chunk_parser.add_argument("text")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int)
    semantic_chunk_parser.add_argument("--overlap", type=int, default=4)
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
            text = (args.text).split()
            chunk_size = args.chunk_size
            overlap = args.overlap
            step = chunk_size - overlap
            chunks = [
                " ".join(text[i: chunk_size + i])
                for i in range(0, len(text), step) if i + overlap  < len(text)
            ]

            print(f"Chunking {len(args.text)} characters")
            for n, chunk in enumerate(chunks, 1):
                print(f"{n}. {chunk.strip()}")
                
        case "semantic_chunk":
            text = args.text
            text_splited = re.split(r"(?<=[.!?])\s+", text)
            print(text_splited)
            size = args.max_chunk_size
            overlap = args.overlap
            step = abs(size - overlap)
            print(step)
            chunks = [
                " ".join(text_splited[i: size + i])
                for i in range(0, len(text_splited), step)
            ]
            
            print(f"Semantically chunking {len(args.text)} characters")
            for n, chunk in enumerate(chunks, 1):
                print(f"{n}. {chunk.strip()}")
            
if __name__ == "__main__":
    main()
