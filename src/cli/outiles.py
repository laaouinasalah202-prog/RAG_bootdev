import json
import string
from nltk.stem import PorterStemmer 
from dotenv import load_dotenv
from openai import OpenAI
import os

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
    return text_tokenized

def load_movie():
    with open("data/movies.json") as d:
        try:
            data = json.load(d)
        except Exception as e:
            print(e)
        return data

def prepare_model():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return client