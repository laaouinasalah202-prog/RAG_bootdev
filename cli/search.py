import json
import string
from nltk.stem import PorterStemmer 


def stopwrd(word):
    with open("data/stop_word.txt") as fd:
        line = fd.readline()
        while line:
            if word+"\n" == line:
                return True
            line = fd.readline()
        return False

def simple_search(query):
    with open("data/movies.json") as d:
        try:
            data = json.load(d)
        except Exception as e:
            print(e)

        resutl = []
        x, y, z = "", "", f"{string.punctuation}"
        mytable = str.maketrans(x,y,z)
        stemmer = PorterStemmer()
        
        query_list = query.split()
        query_list = [wrd.translate(mytable) for wrd in query_list if not stopwrd(wrd.translate(mytable))]
        query_list = [stemmer.stem(wrd) for wrd in query_list]
        
        for movie in data["movies"]:
            for wrd in query_list:

                title_list = movie["title"].split()
                title_listpunc = [wrd.translate(mytable)   for wrd in title_list]
                title_listlower = [wrd.lower() for wrd in title_listpunc]

                if wrd.lower() in title_listlower:
                    resutl.append(movie["title"])
        
        for num, title in enumerate(resutl):
            print(f"{num}. {title}")

