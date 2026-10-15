import argparse
import enum
from lib.semantic_search import SemanticSearch, embed_text, verify_model, verify_embeddings, embed_query_text, semantic_chunk, ChunkedSemanticSearch, load_movies
import json
import re


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")
    embed_chunk_parser = subparsers.add_parser("embed_chunks")
    chunk_parser = subparsers.add_parser("chunk")
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk")
    search_chunked_parser = subparsers.add_parser("search_chunked")
    search_chunked_parser.add_argument("query", type=str)
    search_chunked_parser.add_argument("--limit", type=int, default=5)
    semantic_chunk_parser.add_argument("text")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4)
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0)
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
            semantic_search = SemanticSearch("all-MiniLM-L6-v2")
            movies = load_movies()
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
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
             
            print(f"Semantically chunking {len(args.text)} characters")
            for n, chunk in enumerate(chunks, 1):
                print(f"{n}. {chunk.strip()}")

        case "embed_chunks":
            movies = load_movies()
            semantic_chunk_search = ChunkedSemanticSearch("all-MiniLM-L6-v2")
            embeddings = semantic_chunk_search.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            movies = load_movies
            semantic_chunk_search = ChunkedSemanticSearch("all-MiniLM-L6-v2")
            res = semantic_chunk_search.search_chunks(args.query, args.limit)

            for i, r in enumerate(res, 1):
                print(f'\n{i}. {r["title"]} (score: {r["score"]:.4f})')
                print(f'   {r["document"]}...')
    
if __name__ == "__main__":
    main()
