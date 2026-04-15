from textblob import TextBlob

def get_textblob_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)

    # Get polarity score (-1.0 to 1.0)
    # > 0 is Positive, < 0 is Negative, 0 is Neutral
    polarity = blob.sentiment.polarity

    # Get subjectivity score (0.0 to 1.0)
    # 0 is objective (fact), 1 is subjective (opinion)
    subjectivity = blob.sentiment.subjectivity

    # Logic to classify the label
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, polarity, subjectivity


test_sentences = [
    "I really adore this new smartphone! It's fantastic.",
    "This service is absolutely terrible, the worst I've ever had.",
    "London usually has a lot of cloudy weather.",
    "The film wasn’t too bad, actually it turned out pretty good.",
    "I am very annoyed with how slow the delivery process is."
]

print(f"{'SENTIMENT':<12} | {'POLARITY':<10} | {'SUBJ.':<7} | {'TEXT'}")
print("-" * 80)

for sentence in test_sentences:
    label, pol, subj = get_textblob_sentiment(sentence)
    print(f"{label:<12} | {pol:<10.2f} | {subj:<7.2f} | {sentence}")
