# ============================================================
# ReviewSense — NLP Module
# Sentiment Analysis of Movie Reviews using NLP
# Dataset: IMDb 50K Movie Reviews
# Covers: Tokenization, Stop words, Lemmatization, POS Tagging,
#         Chunking, NER, Text Similarity, Classification, EDA
# ============================================================

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# 1. LOAD IMDB DATASET
# ------------------------------------------------------------
df = pd.read_csv('IMDB Dataset.csv')
df = df.sample(10000, random_state=42).reset_index(drop=True)
df['text'] = df['review']       # rename for consistency with pipeline
# sentiment column already has 'positive'/'negative'

print("Dataset shape:", df.shape)
print(df[['text', 'sentiment']].head())


# ------------------------------------------------------------
# 2. PREPROCESSING (Labs 2, 3)
# ------------------------------------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.encode('ascii', 'ignore').decode('ascii')   # script validation
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)                        # IMDb HTML tags (<br />)
    text = re.sub(r'http\S+|www\S+', '', text)                # URLs
    text = re.sub(r'[^a-z\s]', ' ', text)                     # punctuation/numbers
    tokens = word_tokenize(text)                              # tokenization
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]        # lemmatization
    return " ".join(tokens)

sample = df['text'].iloc[0]
print("\nORIGINAL:", sample[:300])
print("CLEANED :", preprocess(sample)[:300])

df['clean_text'] = df['text'].apply(preprocess)


# ------------------------------------------------------------
# 3. EDA — WORD CLOUDS (Lab 10)
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
df['sentiment'].value_counts().plot(kind='bar', color=['green','red'])
plt.title('Review Sentiment Distribution')
plt.xlabel('Sentiment'); plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('sentiment_distribution.png', dpi=100)
plt.show()

pos_text = " ".join(df[df['sentiment']=='positive']['clean_text'])
wc_pos = WordCloud(width=800, height=400, background_color='white').generate(pos_text)
plt.figure(figsize=(10,5))
plt.imshow(wc_pos, interpolation='bilinear')
plt.axis('off'); plt.title('Positive Reviews — Word Cloud')
plt.tight_layout(); plt.savefig('wordcloud_positive.png', dpi=100); plt.show()

neg_text = " ".join(df[df['sentiment']=='negative']['clean_text'])
wc_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(neg_text)
plt.figure(figsize=(10,5))
plt.imshow(wc_neg, interpolation='bilinear')
plt.axis('off'); plt.title('Negative Reviews — Word Cloud')
plt.tight_layout(); plt.savefig('wordcloud_negative.png', dpi=100); plt.show()


# ------------------------------------------------------------
# 4. POS TAGGING — OPINION WORDS (Lab 6)
# ------------------------------------------------------------
def extract_opinion_words(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    return " ".join([w for w, t in tagged if t.startswith('JJ') or t.startswith('RB')])

demo = df['clean_text'].iloc[0]
print("\nCleaned:", demo[:200])
print("Opinion words:", extract_opinion_words(demo))


# ------------------------------------------------------------
# 5. CHUNKING — NOUN PHRASES (Lab 7)
# ------------------------------------------------------------
def extract_chunks(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    grammar = r"NP: {<DT>?<JJ>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(tagged)
    chunks = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        chunks.append(" ".join(w for w, t in subtree.leaves()))
    return chunks

print("\n--- Chunking Example ---")
print("Noun phrases:", extract_chunks(demo)[:10])


# ------------------------------------------------------------
# 6. NER — NAMED ENTITIES (Lab 8)
# ------------------------------------------------------------
def extract_entities(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tree = ne_chunk(tagged)
    entities = []
    for subtree in tree:
        if isinstance(subtree, Tree):
            entity = " ".join([w for w, t in subtree.leaves()])
            entities.append((entity, subtree.label()))
    return entities

print("\n--- NER Examples ---")
for i in [0, 3, 6]:
    raw = df['text'].iloc[i]
    print(f"\nReview: {raw[:100]}...")
    print(f"Entities: {extract_entities(raw)}")


# ------------------------------------------------------------
# 7. TEXT SIMILARITY (Lab 9)
# ------------------------------------------------------------
sample_reviews = df['clean_text'].head(5).tolist()
tfidf_sim = TfidfVectorizer()
matrix = tfidf_sim.fit_transform(sample_reviews)
sim = cosine_similarity(matrix)

print("\n--- Text Similarity Matrix (first 5 reviews) ---")
print(np.round(sim, 2))


# ------------------------------------------------------------
# 8. SENTIMENT CLASSIFICATION (Lab 12)
# ------------------------------------------------------------
X = df['clean_text']
y = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative','Positive']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Neg','Pos'], yticklabels=['Neg','Pos'])
plt.title(f'Confusion Matrix — Accuracy: {acc:.2%}')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=100); plt.show()


# ------------------------------------------------------------
# 9. LIVE PREDICTION (Demo for Viva)
# ------------------------------------------------------------
def predict_sentiment(review):
    clean = preprocess(review)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
    confidence = prob[pred] * 100
    return {
        "review": review,
        "sentiment": label,
        "confidence": f"{confidence:.1f}%",
        "opinion_words": extract_opinion_words(clean),
        "noun_phrases": extract_chunks(clean),
        "entities": extract_entities(review)
    }

test_reviews = [
    "This movie was absolutely brilliant! The acting was superb and the story kept me on the edge of my seat.",
    "Terrible film. The plot made no sense and the acting was wooden. Complete waste of time.",
    "It was okay, not great but not bad either. Some good scenes but mostly forgettable.",
    "A masterpiece of cinema. The cinematography was stunning and the performances were flawless.",
]

print("\n" + "="*60)
print("LIVE SENTIMENT PREDICTION")
print("="*60)
for t in test_reviews:
    r = predict_sentiment(t)
    print(f"\nReview   : {r['review'][:100]}...")
    print(f"Sentiment: {r['sentiment']}  (Confidence: {r['confidence']})")
    print(f"Opinion  : {r['opinion_words']}")
    print(f"Chunks   : {r['noun_phrases'][:5]}")
    print(f"Entities : {r['entities']}")


# ------------------------------------------------------------
# 10. SAVE MODEL
# ------------------------------------------------------------
import pickle
with open('review_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('tfidf_review_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("\nModel saved: review_model.pkl, tfidf_review_vectorizer.pkl")