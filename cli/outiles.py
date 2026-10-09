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

def tokenize_text(text):
    x, y, z = "", "", f"{string.punctuation}"
    mytable = str.maketrans(x,y,z)
    stemmer = PorterStemmer()
    text = text.split()
    text = [wrd.translate(mytable) for wrd in text if not stopwrd(wrd.translate(mytable))]
    text_tokenized = [stemmer.stem(wrd) for wrd in text]
    return set(text_tokenized)

def load_movie():
    with open("data/movies.json") as d:
        try:
            data = json.load(d)
        except Exception as e:
            print(e)
        return data
