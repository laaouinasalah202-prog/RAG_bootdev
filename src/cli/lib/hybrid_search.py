import os

from .build import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self) -> None:
        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()
        self.documents = self.idx.load()
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings()
        
    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

    def weighted_search(self, query, limit, alpha) -> list:
        bm25_search = self._bm25_search(query, limit * 500)
        semantic_search = self.semantic_search.search_chunks(query, limit * 500)
        doc_scoremap = {}
        for i in range(min(limit*500, len(bm25_search))):
            bm25_score = normalize_score(bm25_search[i][1], bm25_search[-1][1], bm25_search[0][1])
            bm25_id = bm25_search[i][0]
            semantic_score = normalize_score(semantic_search[i]["score"], semantic_search[-1]["score"], semantic_search[0]["score"])
            semantic_id = semantic_search[i]["id"]

            if doc_scoremap.get(bm25_id):
                doc_scoremap[bm25_id].update({"bm25": bm25_score})
            else:
                doc_scoremap.update({bm25_id: {"bm25": bm25_score}})
            
            if doc_scoremap.get(semantic_id):
                doc_scoremap[semantic_id].update({"semantic_score": semantic_score})
            else:
                doc_scoremap.update({semantic_id: {"semantic_score": semantic_score}})

        for doc in doc_scoremap:
            if doc_scoremap[doc].get("bm25") is None:
                doc_scoremap[doc].update({"bm25": 0})
            if doc_scoremap[doc].get("semantic_score") is None:
                doc_scoremap[doc].update({"semantic_score": 0})
            hb = hybrid_score(doc_scoremap[doc]["bm25"], doc_scoremap[doc]["semantic_score"], alpha)
            doc_scoremap[doc].update({"hybrid_score": hb})
            
        result = sorted(doc_scoremap.items(), key=lambda item: item[1]["hybrid_score"], reverse=True)
        return result[:limit]
    
def normalize_score(score, min_score, max_score):
    if max_score == min_score:
        return 1.0

    return round((score - min_score) / (max_score - min_score), 4)

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score