from outiles import load_movie, tokenize_text
from lib.build import InvertedIndex

def simple_search(query, invert_index):
        query_tokenized = list(tokenize_text(query))
        invert_index.load()
        indexes = []
        for token in query_tokenized:
            if token in invert_index.index :
                indexes += invert_index.index[token]
        indexes = list(set(indexes))
        for num, i in enumerate(indexes):
            print(f'ID {i} {invert_index.docmap[i]["description"]}')