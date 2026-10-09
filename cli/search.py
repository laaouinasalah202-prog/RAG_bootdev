from outiles import load_movie, tokenize_text
from build import InvertedIndex

def simple_search(query):

        # data = load_movie()
        # resutl = []
        query_tokenized = list(tokenize_text(query))
        
        # for movie in data["movies"]:
        #     for wrd in query_tokenized:
        #         title_tokenize = list(tokenize_text(movie["title"]))
        #         if wrd.lower() in title_tokenize:
        #             resutl.append(movie["title"])
        #             print(movie["id"])

        
        # for num, title in enumerate(resutl):
        #     print(f"{num}. {title}")

        invert_index =  InvertedIndex()
        index, docmap = invert_index.load()
        indexes = []
        for token in query_tokenized:
            if token in index:
                indexes += index[token]
            
        for num, i in enumerate(indexes):
            print(f'ID {i} {docmap[i]["description"]}')