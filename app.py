import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

ps=PorterStemmer()

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Newspaper Title Classification")

input_text=st.text_area("Enter the title")

if st.button('Predict'):

    transformed_text=transform_text(input_text)

    vector_input=tfidf.transform([transformed_text])

    result=model.predict(vector_input)[0]

    if result == 0:
        st.header("The Entered Title is related to World News")
    elif result == 1:
        st.header("The Entered Title is related to Sports News")
    elif result == 2:
        st.header("The Entered Title is related to Business News")
    else:
        st.header("The Entered Title is related to Sci/Tech News")
