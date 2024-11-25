from flask import Flask, request, render_template
from joblib import load
import re
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import socket

# Инициализация приложения Flask
app = Flask(__name__)

# Загрузка моделей и объектов
vectorizer_1st = load('/Users/daniil/RuTube/RuTube_classificator/new_model/1st_level/tfidf_vectorizer_1st_level.joblib')
model_1st = load('/Users/daniil/RuTube/RuTube_classificator/new_model/1st_level/stacking_model_1st_level.joblib')
label_encoder_1st = load('/Users/daniil/RuTube/RuTube_classificator/new_model/1st_level/label_encoder_1st_level.joblib')

vectorizer_2nd = load('/Users/daniil/RuTube/RuTube_classificator/new_model/2nd_level/tfidf_vectorizer_2nd_level.joblib')
model_2nd = load('/Users/daniil/RuTube/RuTube_classificator/new_model/2nd_level/random_forest_model_2nd_level.joblib')
label_encoder_2nd = load('/Users/daniil/RuTube/RuTube_classificator/new_model/2nd_level/label_encoder_2nd_level.joblib')

vectorizer_3rd = load('/Users/daniil/RuTube/RuTube_classificator/new_model/3rd_level/tfidf_vectorizer_3rd_level.joblib')
model_3rd = load('/Users/daniil/RuTube/RuTube_classificator/new_model/3rd_level/stacking_model_3rd_level.joblib')
label_encoder_3rd = load('/Users/daniil/RuTube/RuTube_classificator/new_model/3rd_level/label_encoder_3rd_level.joblib')

# Стоп-слова
stop_words = set(stopwords.words('russian'))

# Функции для обработки текста
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

segmenter = Segmenter()
embedding = NewsEmbedding()
morph_tagger = NewsMorphTagger(embedding)
morph_vocab = MorphVocab()

def lemmatize_text(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    return ' '.join([token.lemma for token in doc.tokens])

def preprocess_text(text):
    clean = clean_text(text)
    lemmatized = lemmatize_text(clean)
    return lemmatized

# Функция предсказания
def predict_tags(text):
    preprocessed_text = preprocess_text(text)
    
    X_1st_level = vectorizer_1st.transform([preprocessed_text])
    pred_1st_level = model_1st.predict(X_1st_level)
    tag_1st_level = label_encoder_1st.inverse_transform(pred_1st_level)[0]
    
    text_for_2nd = preprocessed_text + " " + tag_1st_level
    
    X_2nd_level = vectorizer_2nd.transform([text_for_2nd])
    pred_2nd_level = model_2nd.predict(X_2nd_level)
    tag_2nd_level = label_encoder_2nd.inverse_transform(pred_2nd_level)[0]
    
    text_for_3rd = text_for_2nd + " " + tag_2nd_level
    
    X_3rd_level = vectorizer_3rd.transform([text_for_3rd])
    pred_3rd_level = model_3rd.predict(X_3rd_level)
    tag_3rd_level = label_encoder_3rd.inverse_transform(pred_3rd_level)[0]
    
    return {
        "1st Level Tag": tag_1st_level,
        "2nd Level Tag": tag_2nd_level,
        "3rd Level Tag": tag_3rd_level
    }

# Главная страница с формой
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form["text"]
        results = predict_tags(input_text)
        return render_template("result.html", text=input_text, results=results)
    return render_template("index.html")

# Функция для поиска свободного порта
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Привязка к свободному порту
        return s.getsockname()[1]

# Запуск приложения
if __name__ == "__main__":
    port = find_free_port()
    print(f"Сервер запущен на порту: {port}")
    app.run(host="0.0.0.0", port=port, debug=True)





