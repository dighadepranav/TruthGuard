import pickle
import re
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load saved model
with open('review_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_review_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

print("="*60)
print("REVIEWSENSE — LIVE PREDICTION")
print("Type a review (or 'quit' to exit)")
print("="*60)

while True:
    review = input("\nEnter review: ").strip()
    if review.lower() in ('quit', 'exit', 'q'):
        break
    if not review:
        continue
    clean = preprocess(review)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
    print(f"\n  → Sentiment: {label}")
    print(f"  → Confidence: {prob[pred] * 100:.1f}%")

print("\nGoodbye!")