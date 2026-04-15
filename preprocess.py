import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def preprocess(text):
    text = str(text).lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)