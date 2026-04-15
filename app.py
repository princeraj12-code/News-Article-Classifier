import pickle
from preprocess import preprocess

model = pickle.load(open("model/news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def predict(text):
    text = preprocess(text)
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

while True:
    text = input("Enter news (or exit): ")
    if text == "exit":
        break
    print("Category:", predict(text))