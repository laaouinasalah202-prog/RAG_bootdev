
from copyreg import pickle
import os
from pickle import TRUE
from pydoc import text
from sentence_transformers import SentenceTransformer
import numpy as np
from lib.build import InvertedIndex
import json
import re

class SemanticSearch:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.embeddings = None
        self.documents = None
        self.documents_map = {}

    def build_embeddings(self, documents):
        self.documents = documents
        self.documents_map = {doc["id"]:{"description": doc["description"], "title": doc["title"]} for doc in self.documents["movies"]}
        str_docs = [f'{doc["title"]}:{doc["description"]}' for doc in self.documents["movies"]]
        self.embeddings = self.model.encode(str_docs, show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            self.documents = documents
            self.documents_map = {doc["id"]:{"description": doc["description"], "title": doc["title"]} for doc in self.documents["movies"]}
            if len(self.documents) == len(self.embeddings):
                return self.embeddings
        else:
            return self.build_embeddings(documents)

    def search(self, query, limit):
        if not os.path.exists("cache/movie_embeddings.npy"):
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        all_similarity = []
        
        for id, i in enumerate(self.embeddings):
            cosine_sim = cosine_similarity(query_embedding, i)
            all_similarity.append((cosine_sim, id+1))

        sorted_similarity = sorted(all_similarity, key=lambda x: x[0], reverse=True)
        top_limit = [{"cosine": cosine, "title": self.documents_map[doc]["title"], "description": self.documents_map[doc]["description"]} for cosine, doc in sorted_similarity[:5]]
        return top_limit

        
    def generate_embedding(self, text):
        if text.strip() is None:
            raise ValueError("the test given the generate embedding is empty")
        embeddings = self.model.encode(text)
        return embeddings

def semantic_chunk(text, size, overlap):
    text = text.strip()
    text_splited = re.split(r"(?<=[.!?])\s+", text)
    step = size - overlap
    chunks = [
        " ".join(text_splited[i: size + i])
        for i in range(0, len(text_splited), step) if i + overlap < len(text_splited)
    ]
    return chunks


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self) -> np.ndarray:
        self.documents = load_movies()
        self.documents_map = {doc["id"]:{"description": doc["description"], "title": doc["title"]} for doc in self.documents["movies"]}
        all_chunks: list[str] = []
        chunks_metadata: list[dict] = []

        for id, doc in self.documents_map.items():
            doc_chunks = semantic_chunk(doc["description"], 4, 1)
            all_chunks += doc_chunks

            for i in range(len(doc_chunks)):
                chunk_idx = len(all_chunks) + i
                chunks_metadata.append({"movie_idx": id, "chunk_idx": i, "total_chunks": len(doc_chunks)})

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_metadata

        with open("cache/chunk_embeddings.npy", "wb") as f:
            np.save(f, self.chunk_embeddings)

        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": chunks_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self) -> np.ndarray:
        self.documents = load_movies()
        self.documents_map = {doc["id"]:{"description": doc["description"], "title": doc["title"]} for doc in self.documents["movies"]}
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            with open("cache/chunk_embeddings.npy", "rb") as f:
                self.chunk_embeddings = np.load(f)
            with open("cache/chunk_metadata.json") as f:
                self.chunk_metadata = json.load(f)["chunks"]
        else:
            self.chunk_embeddings = self.build_chunk_embeddings()
        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        chunk_score: list[dict] = []
        embeddings = self.load_or_create_chunk_embeddings()

        # calculate the cosine similarity for each chunk preserving the metadata
        for i, emb in enumerate(embeddings):
            cosine_sim = cosine_similarity(query_embedding, emb)
            chunk_score.append({"chunk_idx": i, "movie_idx": self.chunk_metadata[i]["movie_idx"], "cosine": cosine_sim})
        map_bestChunk = {}
        
        # calculate the best score for each movies 
        for n, chunk in enumerate(chunk_score):
            score = map_bestChunk.get(chunk["movie_idx"])
            if score and chunk["cosine"] > score:
                map_bestChunk[chunk["movie_idx"]] = chunk["cosine"]
            if score is None:
                map_bestChunk.update({chunk["movie_idx"]: chunk["cosine"]})
        
        sorted_movies = sorted(
                map_bestChunk.items(),
                key=lambda item: item[1],
                reverse=True
            )
        best_score_limit = sorted_movies[:limit]
        result_dict = [
            {"id": movie[0], "title": self.documents_map[movie[0]]["title"], "document": self.documents_map[movie[0]]["description"][:100], "score": round(movie[1], 3)} 
            for movie in best_score_limit
        ]
        return result_dict
        

def load_movies() -> list[dict]:
    with open("data/movies.json", "r") as f:
        movies = json.load(f)
    return movies

def embed_query_text(query):
    semantic_search = SemanticSearch("all-MiniLM-L6-v2")
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def verify_embeddings():
    semantic_search = SemanticSearch("all-MiniLM-L6-v2")
    movies = load_movies()
    movies_list = [movie for movie in movies["movies"]]
    embeddings = semantic_search.load_or_create_embeddings(movies_list)
    print(f"Number of docs:   {len(semantic_search.documents)}")
    print(
        f"Embeddings shape: {semantic_search.embeddings.shape[0]} vectors in {semantic_search.embeddings.shape[1]} dimensions"
    )
    return embeddings

def verify_model():
    semantic_search = SemanticSearch("all-MiniLM-L6-v2")
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def  embed_text(text):
    sem_search = SemanticSearch("all-MiniLM-L6-v2")
    embedding = sem_search.generate_embedding(text)
    print(f"Text: {text}", type(embedding))
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)