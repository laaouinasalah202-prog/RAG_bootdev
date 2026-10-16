import argparse
from lib.hybrid_search import HybridSearch
import os
from dotenv import load_dotenv
from openai import OpenAI

def prepare_model():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return client

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparser.add_parser("normalize")
    rrf_search_parser = subparser.add_parser("rrf-search")
    rrf_search_parser.add_argument("query")
    rrf_search_parser.add_argument("--limit", default=5, type=int)
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument("-k", default=60, type=int)
    
    
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
                
        case "rrf-search":
            if args.enhance == "spell":
                client = prepare_model()
                content = f"""Fix any spelling errors in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                Preserve punctuation and capitalization unless a change is required for a typo fix.
                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text, nothing else.
                User query: "{args.query}"
                """ 
                messages = [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
                respones = client.chat.completions.create(messages=messages, model="openrouter/free")
                enhanced_query = respones.choices[0].message.content
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")
                args.query = enhanced_query
            if args.enhance == "rewrite":
                client = prepare_model()
                content = f"""Rewrite the user-provided movie search query below to be more specific and searchable.
                
                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep the rewritten query concise (under 10 words)
                - It should be a Google-style search query, specific enough to yield relevant results
                - Don't use boolean logic
                
                Examples:
                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"
                
                If you cannot improve the query, output the original unchanged.
                Output only the rewritten query text, nothing else.
                
                User query: "{args.query}"
                """
                messages = [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
                responses = client.chat.completions.create(messages=messages, model="openrouter/free")
                new_query = responses.choices[0].message.content
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{new_query}'\n")
                args.query = new_query
            if args.enhance == "expand":
                client = prepare_model()
                content = f"""Expand the user-provided movie search query below with related terms.
                
                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                Output only the additional terms; they will be appended to the original query.
                
                Examples:
                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"
                
                User query: "{args.query}"
                """
                messages = [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
                response = client.chat.completions.create(messages=messages, model="openrouter/free")
                expanded_query = response.choices[0].message.content
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{expanded_query}'\n")
            hybrid_search = HybridSearch()
            best_result = hybrid_search.rrf_search(args.query, args.k, args.limit)
            for n, rs in enumerate(best_result, 1):
                print(f'{n}. {hybrid_search.idx.docmap[rs[0]]["title"]}\n'
                f'RRF Score: {rs[1]["rrf"]}\n'
                f'BM25 Rank: {rs[1]["bm25_rank"]}, Semantic Rank: {rs[1]["semantic_rank"]}\n'
                f'{hybrid_search.idx.docmap[rs[0]]["description"][:100]}')
    
if __name__ == "__main__":
    main()
