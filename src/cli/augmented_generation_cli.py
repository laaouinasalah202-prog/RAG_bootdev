import argparse
from lib.hybrid_search import HybridSearch
from outiles import prepare_model

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("query")
    summarize_parser.add_argument("--limit", default=10, type=int)
    citation_parser = subparsers.add_parser("citations")
    citation_parser.add_argument("query")
    citation_parser.add_argument("--limit", default=10, type=int)
    
    args = parser.parse_args()

    match args.command:
        case "rag":
            limit = 10
        
            query = args.query
            hybrid_search = HybridSearch()
            rrf_result = hybrid_search.rrf_search(query, 60, limit)
            client = prepare_model()
            doc_map = hybrid_search.idx.docmap
            docs = [f'title: {doc_map[res[0]]["title"]}, description: {doc_map[res[0]]["description"]}' for res in rrf_result]
            prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
                Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
                Provide a comprehensive answer that addresses the user's query.

                Query: {query}
            
                Documents:
                    {docs}
            
                Answer:"""
            messages = [{
                "role": "user",
                "content": prompt
            }]
            response = client.chat.completions.create(messages=messages, model="openrouter/free")
            print('Search Results:')
            for i in rrf_result:
                print(f'- {doc_map[i[0]]["title"]}')
            print(f'RAG Response:\n{response.choices[0].message.content}')

        case "summarize":
            hybird_search  = HybridSearch()
            doc_map = hybird_search.idx.docmap
            k = 60
            
            rrf_result = hybird_search.rrf_search(args.query, k, args.limit)
            client = prepare_model()
            result = [f'title: {doc_map[res[0]]["title"]}, description: {doc_map[res[0]]["description"]}' for res in rrf_result]
            content = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.
            
            The goal is to provide comprehensive information so that users know what their options are.
            Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
            
            This should be tailored to Webflyx users. Webflyx is a movie streaming service.
            
            Query: {args.query}
            
            Search results:
            {result}
            
            Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            response = client.chat.completions.create(messages=messages, model="openrouter/free")
            print('Search Results:')
            for i in rrf_result:
                print(f'- {doc_map[i[0]]["title"]}')
            print(f'LLM Summary:\n{response.choices[0].message.content}')

        case "citations":
            hybrid_search = HybridSearch()
            doc_map = hybrid_search.idx.docmap
            query, k = args.query, 60
            rrf_res = hybrid_search.rrf_search(query, k, limit=args.limit)
            client = prepare_model()
            result = [f'title: {doc_map[res[0]]["title"]}, description: {doc_map[res[0]]["description"]}' for res in rrf_res]
            content = prompt = f"""Answer the query below and give information based on the provided documents.
            
            The answer should be tailored to users of Webflyx, a movie streaming service.
            If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.
            
            Query: {query}
            
            Documents:
            {result}
            
            Instructions:
            - Provide a comprehensive answer that addresses the query
            - Cite sources in the format [1], [2], etc. when referencing information
            - If sources disagree, mention the different viewpoints
            - If the answer isn't in the provided documents, say "I don't have enough information"
            - Be direct and informative
            
            Answer:"""
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            response = client.chat.completions.create(messages=messages, model="openrouter/free")
            print('Search Results:')
            for i in rrf_res:
                print(f'- {doc_map[i[0]]["title"]}')
            print(f'LLM Answer:\n{response.choices[0].message.content}')
            
        case _:
            parser.print_help()
            

if __name__ == "__main__":
    main()
