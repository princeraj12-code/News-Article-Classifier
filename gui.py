import tkinter as tk
from tkinter import messagebox
import pickle
from preprocess import preprocess

# Load model
model = pickle.load(open("model/news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def predict_news():
    text = entry.get("1.0", tk.END).strip()
    
    if text == "":
        messagebox.showwarning("Warning", "Please enter some text")
        return
    
    text_clean = preprocess(text)
    vector = vectorizer.transform([text_clean])
    result = model.predict(vector)[0]
    
    output_label.config(text=f"Category: {result}")

# Create window
root = tk.Tk()
root.title("News Classifier")
root.geometry("500x400")

# Title
title = tk.Label(root, text="News Article Classifier", font=("Arial", 16))
title.pack(pady=10)

# Text input
entry = tk.Text(root, height=10, width=50)
entry.pack(pady=10)

# Predict button
btn = tk.Button(root, text="Predict Category", command=predict_news)
btn.pack(pady=10)

# Output label
output_label = tk.Label(root, text="", font=("Arial", 14))
output_label.pack(pady=20)

# Run app
root.mainloop()