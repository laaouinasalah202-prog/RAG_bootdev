
from copyreg import pickle
import os
from pickle import TRUE
from pydoc import text
from sentence_transformers import SentenceTransformer
import numpy as np
from lib.build import InvertedIndex
import json

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.documents_map = {}

    def build_embeddings(self, documents):
        self.documents = documents
        self.documents_map = {doc["id"]:{"description": doc["description"], "title": doc["title"]} for doc in self.documents["movies"]}
        str_docs = [f'{doc["title"]}:{doc["description"]}' for doc in self.documents]
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
            self.build_embeddings(documents)

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

def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    with open("data/movies.json", "r") as f:
        movies = json.load(f)
    movies_list = [movie for movie in movies["movies"]]
    semantic_search.load_or_create_embeddings(movies_list)
    print(f"Number of docs:   {len(semantic_search.documents)}")
    print(
        f"Embeddings shape: {semantic_search.embeddings.shape[0]} vectors in {semantic_search.embeddings.shape[1]} dimensions"
    )

def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def  embed_text(text):
    sem_search = SemanticSearch()
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