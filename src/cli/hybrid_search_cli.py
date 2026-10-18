import argparse
from audioop import cross
from time import sleep
from typing_extensions import Optional
from lib.hybrid_search import HybridSearch, rrf_score
import os
from outiles import prepare_model
import json
from sentence_transformers import CrossEncoder



def llm_evaluate(query, formatted_results):
    client = prepare_model()
    content = f"""Rate how relevant each result is to this query on a 0-3 scale:
    
    Query: "{query}"
    
    Results:
    {chr(10).join(formatted_results)}
    
    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant
    
    Do NOT give any numbers other than 0, 1, 2, or 3.
    
    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:
    
    [2, 0, 3, 2, 0, 1]"""
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    response = client.chat.completions.create(messages=messages, model="openrouter/free")
    resp =  json.loads(response.choices[0].message.content)
    return resp
    
def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparser.add_parser("normalize")
    rrf_search_parser = subparser.add_parser("rrf-search")
    rrf_search_parser.add_argument("query")
    rrf_search_parser.add_argument("--rerank-method", choices=["individual", "batch", "cross_encoder"])
    rrf_search_parser.add_argument("--limit", default=5, type=int)
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    
    rrf_search_parser.add_argument("-k", default=60, type=int)
    rrf_search_parser.add_argument("--evaluate", help="evaluate the search result", action="store_true")
    
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
            if args.rerank_method:
                args.limit *= 5
            hybrid_search = HybridSearch()
            best_result = hybrid_search.rrf_search(args.query, args.k, args.limit)
            if not args.rerank_method:
                for n, rs in enumerate(best_result, 1):
                    print(f'{n}. {hybrid_search.idx.docmap[rs[0]]["title"]}\n'
                    f'RRF Score: {rs[1]["rrf"]}\n'
                    f'BM25 Rank: {rs[1]["bm25_rank"]}, Semantic Rank: {rs[1]["semantic_rank"]}\n'
                    f'{hybrid_search.idx.docmap[rs[0]]["description"][:100]}')

            if args.rerank_method == "individual":
                client = prepare_model()
                doc_map = hybrid_search.idx.docmap
                scores = []
                for doc in best_result[:3]:
                    content = f"""Rate how well this movie matches the search query.
                    
                    Query: "{args.query}"
                    Movie: {doc_map[doc[0]].get("title", "")} - {doc_map[doc[0]].get("document", "")}
                    
                    Consider:
                    - Direct relevance to query
                    - User intent (what they're looking for)
                    - Content appropriateness
                    
                    Rate 0-10 (10 = perfect match).
                    Output ONLY the number in your response, no other text or explanation.
                    
                    Score:"""
                    message = [
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
                    respones = client.chat.completions.create(messages=message, model="openrouter/free")
                    scores.append(respones.choices[0].message.content)
                    sleep(3)

                    print('Re-ranking top 3 results using individual method...')           
                    print(f'Reciprocal Rank Fusion Results for {args.query} (k={args.k}):')
                    for n, i in enumerate(scores, 1):
                        print(f'''{n}. {doc_map[best_result[n-1][0]]["title"]}
                            Re-rank Score: {i}
                            RRF Score: {best_result[n-1][1]["rrf"]}
                            BM25 Rank: {best_result[n-1][1]["bm25_rank"]}, Semantic Rank: {best_result[n-1][1]["semantic_rank"]}
                            {doc_map[best_result[n-1][0]]["description"]}
                            ''')
            if args.rerank_method == "batch":
                client = prepare_model()
                doc_map = hybrid_search.idx.docmap
                doc_list_str = [
                    f'{doc_map[doc[0]].get("title", "")} , description: {doc_map[doc[0]].get("document", "")}'
                    for doc in best_result
                ]
                
                content = f"""Rank the movies listed below by relevance to the following search query.
                Query: "{args.query}"
                
                Movies:
                {doc_list_str}
                
                Return the movie IDs in order of relevance, best match first.
                
                Your response must be a raw JSON array of integers.
                Do not wrap the JSON in Markdown. Do not use a ```json code block.
                Do not include any explanatory text.
                
                For example:
                [75, 12, 34, 2, 1]
                
                Ranking:"""
                message = [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
                respones = client.chat.completions.create(messages=message, model="openrouter/free")
                scores = respones.choices[0].message.content
                scores = json.loads(scores)
                print("e")
                print('Re-ranking top 3 results using individual method...')           
                print(f'Reciprocal Rank Fusion Results for {args.query} (k={args.k}):')
                for n, i in enumerate(scores[:3], 1):
                    print(f'''{n}. {doc_map[best_result[n-1][0]]["title"]}
                        Re-rank Score: {i}
                        RRF Score: {best_result[n-1][1]["rrf"]}
                        BM25 Rank: {best_result[n-1][1]["bm25_rank"]}, Semantic Rank: {best_result[n-1][1]["semantic_rank"]}
                        {doc_map[best_result[n-1][0]]["description"]}
                        ''')
                        
            if args.rerank_method == "cross_encoder":
                doc_map = hybrid_search.idx.docmap
                pairs = [[args.query, f'{doc_map[doc[0]].get("title", "")} - {doc_map[doc[0]].get("title", "")}'] for doc in best_result]
                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                scores = cross_encoder.predict(pairs)
                for n, i in enumerate(scores, 1):
                    print(f'''{n}. {doc_map[best_result[n-1][0]]["title"]}
                        Cross Encoder Score: {i}
                        RRF Score: {best_result[n-1][1]["rrf"]}
                        BM25 Rank: {best_result[n-1][1]["bm25_rank"]}, Semantic Rank: {best_result[n-1][1]["semantic_rank"]}
                        {doc_map[best_result[n-1][0]]["description"]}
                        ''')
                
            if args.evaluate:
                formated_result = [f'title : {hybrid_search.idx.docmap[res[0]]["title"]}, description {hybrid_search.idx.docmap[res[0]]["description"]}' for  res in best_result]
                llm_rank = llm_evaluate(args.query, formated_result)
                for i, n in enumerate(best_result):
                    print(f'{i+1}. {hybrid_search.idx.docmap[n[0]]["title"]}: {llm_rank[i]}/3')
if __name__ == "__main__":
    main()
