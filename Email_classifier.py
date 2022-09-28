import streamlit as sr
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import re
import numpy as np

ps=PorterStemmer()
stopwords=stopwords.words("english")

cv=pickle.load(open("vectorizer.pkl","rb"))
Bnb=pickle.load(open("model.pkl","rb"))

def text_to_feature(text):
    words=[]
    
    for i in re.sub("[^a-z]"," ",text.lower()).split():

        if i not in stopwords:
            words.append(ps.stem(i))

    return " ".join(words)
        
        
sr.title("Email Spam Classifier")
input_data=sr.text_area("Enter Message")

new_text=text_to_feature(input_data)

if sr.button("Predict"):
    x=cv.transform([new_text]).toarray()

    Predict=Bnb.predict(x)[0]

    if Predict==1:
        sr.text("Spam")

    else:
        sr.text("Not Spam")
