from asyncio.streams import protocols
from concurrent.futures.process import _ExceptionWithTraceback
from copyreg import pickle
from logging import exception
import pickle
from nltk.classify.senna import path
import os
from outiles import load_movie
from outiles import tokenize_text

class InvertedIndex:
    def __init__(self):
        self.index  = {}
        self.docmap = {}
    
    def __add_document(self, doc_id, text):
        self.docmap[doc_id] = {
            "title": text[0],
            "description": text[1]
        }
    
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
            data_tokenzed = list(tokenize_text(text["description"] + text["title"]))
            for token in data_tokenzed:
                if token not in self.index:
                    self.index[token] = []
                self.index[token].append(id)

            
                
    def save(self):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        import pickle
        try:
            with open("cache/index.pkl", "wb") as f:
                pickle.dump(self.index, f, protocol=None) 
            with open("cache/docmap.pkl", "wb") as f:
                pickle.dump(self.docmap, f, protocol=None)
        except pickle.PickleError as e:
            print(e)
            
    def load(self):
        try:
            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)   
            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f) 
        except pickle.PickleError as e:
            print("and error occure when loading the index and docmap form the cache file")
        return self.index, self.docmap