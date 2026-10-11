# TweetSense — NLP Module (No spaCy version)
# Sentiment Analysis + POS Tagging + Chunking + NER (NLTK only)

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Download all required NLTK resources
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
# 1. LOAD DATASET
# ------------------------------------------------------------
df = pd.read_csv(
    'training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    header=None,
    names=['sentiment', 'id', 'date', 'query', 'user', 'text']
)

df = df.sample(50000, random_state=42).reset_index(drop=True)
df['sentiment'] = df['sentiment'].map({0: 'negative', 4: 'positive'})

print("Dataset shape:", df.shape)
print(df[['text', 'sentiment']].head())


# ------------------------------------------------------------
# 2. PREPROCESSING (Labs 2, 3)
# ------------------------------------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.encode('ascii', 'ignore').decode('ascii')  # script validation
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)              # URLs
    text = re.sub(r'@\w+', '', text)                        # mentions
    text = re.sub(r'#', '', text)                           # hashtag symbol
    text = re.sub(r'[^a-z\s]', ' ', text)                   # punctuation/numbers
    tokens = word_tokenize(text)                            # tokenization
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]      # lemmatization
    return " ".join(tokens)

sample = df['text'].iloc[0]
print("ORIGINAL:", sample)
print("CLEANED :", preprocess(sample))

df['clean_text'] = df['text'].apply(preprocess)


# ------------------------------------------------------------
# 3. EDA — WORD CLOUDS (Lab 10)
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
df['sentiment'].value_counts().plot(kind='bar', color=['green','red'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment'); plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('sentiment_distribution.png', dpi=100)
plt.show()

pos_text = " ".join(df[df['sentiment']=='positive']['clean_text'])
wc_pos = WordCloud(width=800, height=400, background_color='white').generate(pos_text)
plt.figure(figsize=(10,5))
plt.imshow(wc_pos, interpolation='bilinear')
plt.axis('off'); plt.title('Positive Tweets — Word Cloud')
plt.tight_layout(); plt.savefig('wordcloud_positive.png', dpi=100); plt.show()

neg_text = " ".join(df[df['sentiment']=='negative']['clean_text'])
wc_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(neg_text)
plt.figure(figsize=(10,5))
plt.imshow(wc_neg, interpolation='bilinear')
plt.axis('off'); plt.title('Negative Tweets — Word Cloud')
plt.tight_layout(); plt.savefig('wordcloud_negative.png', dpi=100); plt.show()


# ------------------------------------------------------------
# 4. POS TAGGING — OPINION WORDS (Lab 6)
# ------------------------------------------------------------
def extract_opinion_words(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    return " ".join([w for w, t in tagged if t.startswith('JJ') or t.startswith('RB')])

demo = df['clean_text'].iloc[0]
print("\nCleaned:", demo[:150])
print("Opinion words:", extract_opinion_words(demo))


# ------------------------------------------------------------
# 5. CHUNKING — NOUN PHRASE EXTRACTION (Lab 7)
# ------------------------------------------------------------
def extract_chunks(text):
    """Extract noun phrases using a simple regex chunk grammar."""
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
chunks = extract_chunks(demo)
print("Noun phrases:", chunks[:10])


# ------------------------------------------------------------
# 6. NER — NAMED ENTITY RECOGNITION using NLTK (Lab 8)
#    (Replaces spaCy — uses POS tagging + ne_chunk internally)
# ------------------------------------------------------------
def extract_entities(text):
    """Extract PERSON, ORGANIZATION, GPE entities using NLTK ne_chunk."""
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tree = ne_chunk(tagged)
    entities = []
    for subtree in tree:
        if isinstance(subtree, Tree):
            entity = " ".join([word for word, tag in subtree.leaves()])
            label = subtree.label()  # PERSON, ORGANIZATION, GPE, etc.
            entities.append((entity, label))
    return entities

print("\n--- NER Examples (NLTK ne_chunk) ---")
for i in [0, 5, 10]:
    raw = df['text'].iloc[i]
    ents = extract_entities(raw)
    print(f"\nTweet: {raw[:100]}")
    print(f"Entities: {ents}")


# ------------------------------------------------------------
# 7. TEXT SIMILARITY (Lab 9)
# ------------------------------------------------------------
sample_tweets = df['clean_text'].head(5).tolist()
tfidf_sim = TfidfVectorizer()
matrix = tfidf_sim.fit_transform(sample_tweets)
sim = cosine_similarity(matrix)

print("\n--- Text Similarity Matrix (first 5 tweets) ---")
print(np.round(sim, 2))


# ------------------------------------------------------------
# 8. SENTIMENT CLASSIFICATION (Lab 12 — Main)
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
def predict_sentiment(tweet):
    clean = preprocess(tweet)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
    confidence = prob[pred] * 100
    return {
        "tweet": tweet,
        "sentiment": label,
        "confidence": f"{confidence:.1f}%",
        "opinion_words": extract_opinion_words(clean),
        "noun_phrases": extract_chunks(clean),
        "entities": extract_entities(tweet)
    }

test_tweets = [
    "I absolutely love the new iPhone! Camera is amazing.",
    "Worst flight ever. Delayed 6 hours, no apology.",
    "Apple released a new update in California yesterday.",
    "The food was okay but service was very slow.",
]

print("\n" + "="*60)
print("LIVE SENTIMENT PREDICTION")
print("="*60)
for t in test_tweets:
    r = predict_sentiment(t)
    print(f"\nTweet    : {r['tweet']}")
    print(f"Sentiment: {r['sentiment']}  (Confidence: {r['confidence']})")
    print(f"Opinion  : {r['opinion_words']}")
    print(f"Chunks   : {r['noun_phrases'][:5]}")
    print(f"Entities : {r['entities']}")


# ------------------------------------------------------------
# 10. SAVE MODEL
# ------------------------------------------------------------
import pickle
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("\nModel saved.")