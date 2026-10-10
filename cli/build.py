from asyncio.streams import protocols
from concurrent.futures.process import _ExceptionWithTraceback
from copyreg import pickle
from logging import exception
import pickle
from nltk.classify.senna import path
import os
from outiles import load_movie, tokenize_text
import math

class InvertedIndex:
    def __init__(self):
        self.index  = {}
        self.docmap = {}
        self.term_frequencies = {}

    def __add_document(self, doc_id, text):
        self.docmap[doc_id] = {
            "title": text[0],
            "description": text[1]
        }
        
    def calculate_term_frequencie(self, token, id):
        if token not in self.term_frequencies:
            self.term_frequencies[token] = {}
        if id not in self.term_frequencies[token]:
            self.term_frequencies[token].update({id: 0})
        self.term_frequencies[token][id] += 1
        
    def get_documents(self, term):
        pass

    def build(self):
        data = load_movie()
        # building the docmap
        for movie in data["movies"]:
            text = (movie["description"], movie["title"])
            self.__add_document(movie["id"], text)

        # building the index
        for id, text in (self.docmap).items():
            data_tokenzed = tokenize_text(text["description"] + text["title"])
            for token in data_tokenzed:
                if token not in self.index:
                    self.index[token] = []
                self.index[token].append(id)
                self.calculate_term_frequencie(token, id)
        
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