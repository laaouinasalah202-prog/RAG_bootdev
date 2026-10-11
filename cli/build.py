from asyncio.streams import protocols
from copyreg import pickle
from logging import exception
import pickle
from pydoc import doc
from tkinter import LabelFrame
from nltk.classify.senna import path
import os
from outiles import load_movie, tokenize_text
import math

class InvertedIndex:
    BM25_K1 = 1.5
    BM25_B = 0.75
    def __init__(self):
        self.index  = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}
        CACHE_DIR = "cache"
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        
    def __add_document(self, doc_id, text):
        self.docmap[doc_id] = {
            "title": text[0],
            "description": text[1]
        }
        
    def calculate_term_frequencie(self, token, doc_id):
        if token not in self.term_frequencies:
            self.term_frequencies[token] = {}
        if doc_id not in self.term_frequencies[token]:
            self.term_frequencies[token].update({doc_id: 0})
        self.term_frequencies[token][doc_id] += 1
        
    def get_documents(self, term):
        pass

    def build(self):
        data = load_movie()
        # building the docmap
        for movie in data["movies"]:
            text = (movie["title"], movie["description"])
            self.__add_document(movie["id"], text)
        # building the index
        for doc_id, text in (self.docmap).items():
            data_tokenzed = tokenize_text(text["description"] + text["title"])
            self.doc_lengths.update({doc_id: len(data_tokenzed)})
            for token in data_tokenzed:
                if token not in self.index:
                    self.index[token] = []
                self.index[token].append(doc_id)
                self.calculate_term_frequencie(token, doc_id)

    def save(self):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        import pickle
        try:
            with open("cache/index.pkl", "wb") as f:
                pickle.dump(self.index, f, protocol=None) 
            with open("cache/docmap.pkl", "wb") as f:
                pickle.dump(self.docmap, f, protocol=None)
            with open("cache/term_frequencies.pkl", "wb") as f:
                pickle.dump(self.term_frequencies, f, protocol=None)
            with open(self.doc_lengths_path, "wb") as f:
                pickle.dump(self.doc_lengths, f, protocol=None)
        except pickle.PickleError as e:
            print(e)
            
    def load(self):
        try:
            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)   
            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f) 
            with open("cache/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
        except pickle.PickleError as e:
            print("and error occure when loading the index and docmap form the cache file")

    def term_tokenize(self, term) -> list:
        token = tokenize_text(term)
        if not token:
            raise ValueError("the term dose not match in any token")
        return token

    def get_tf(self, doc_id, term) -> int:
        token = self.term_tokenize(term)[0]
        if doc_id in self.term_frequencies[token]:
            return self.term_frequencies[token][doc_id]
        else:
            return 0

    def calculate_idf(self, term) -> float:
        token = self.term_tokenize(term)[0]
        term_match_doc_count =  len(self.term_frequencies[token])
        total_doc_count = len(self.docmap)
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def calculate_tfidf(self, term, doc_id) -> float:
        return self.calculate_idf(term) * self.get_tf(doc_id, term)

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        token = self.term_tokenize(term)[0]
        if token in self.term_frequencies:
            df = len(self.term_frequencies[token])
        else:
            raise KeyError("token not found")
        idf_bm25 = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return idf_bm25

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        lenght_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)

        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * lenght_norm)
        return bm25_tf

    def  __get_avg_doc_length(self) -> float:
        all_doc_len = sum([lenght for doc_id, lenght in self.doc_lengths.items()])
        total_doc = len(self.docmap)
        avg_doc_lenght = all_doc_len / total_doc
        return avg_doc_lenght

    def bm25(self, doc_id, term) -> float:
        bm25 = self.get_bm25_idf(term) * self.get_bm25_tf(doc_id, term)
        return bm25

    def bm25_search(self, query, limit=5):
        tokenized_query = tokenize_text(query)
        doc_bm25 = {}
        for doc_id in self.docmap:
            for token in tokenized_query:
                if doc_id not in doc_bm25:
                    doc_bm25.update({doc_id: 0})
                bm25_score = self.bm25(doc_id, token)
                doc_bm25[doc_id] += bm25_score

        doc_map = sorted(doc_bm25.items(), key=lambda doc_bm25: doc_bm25[1], reverse=True)
        return doc_map[:limit]