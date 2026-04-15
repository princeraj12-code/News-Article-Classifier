import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from preprocess import preprocess

# Load dataset (IMPORTANT: tab separator)
df = pd.read_csv("dataset/bbc-news-data.csv", sep='\t')

print("Columns:", df.columns)

# Remove empty rows
df = df.dropna()

# Combine title + content
df['text'] = df['title'] + " " + df['content']

X = df['text']
y = df['category']

# Clean text
X = X.apply(preprocess)

# Convert text → numbers
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
pickle.dump(model, open("model/news_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("✅ Training complete!")