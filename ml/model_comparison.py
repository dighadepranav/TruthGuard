# ============================================================
# TweetSense — ML Module
# Compare 6+ classifiers + Feature Importance + Clustering + LDA
# ============================================================

import pandas as pd
import numpy as np
import re
import os
import nltk
import matplotlib
matplotlib.use('TkAgg')   # ensures plot windows show up
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.metrics import accuracy_score, silhouette_score


# ------------------------------------------------------------
# 1. LOAD + PREPROCESS
# ------------------------------------------------------------
CSV_PATH = os.path.join('..', 'nlp', 'training.1600000.processed.noemoticon.csv')

df = pd.read_csv(
    CSV_PATH,
    encoding='latin-1',
    header=None,
    names=['sentiment', 'id', 'date', 'query', 'user', 'text']
)
df = df.sample(30000, random_state=42).reset_index(drop=True)
df['sentiment'] = df['sentiment'].map({0: 'negative', 4: 'positive'})

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

print("Preprocessing 30,000 tweets...")
df['clean_text'] = df['text'].apply(preprocess)

X_text = df['clean_text']
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# Feature extraction
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2)
X = tfidf.fit_transform(X_text)

# Manual split (avoids blocked sklearn.model_selection)
split = int(0.8 * X.shape[0])
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}\n")


# ------------------------------------------------------------
# 2. TRAIN 6 MODELS + COMPARE
# ------------------------------------------------------------
models = {
    "Naive Bayes":          MultinomialNB(),
    "Logistic Regression":  LogisticRegression(max_iter=1000),
    "Decision Tree (CART)": DecisionTreeClassifier(max_depth=20, random_state=42),
    "Linear SVM":           LinearSVC(max_iter=2000),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

# XGBoost (optional — only if installed)
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        eval_metric='logloss', random_state=42
    )
except ImportError:
    print("⚠️ XGBoost not installed — skipping. Run: pip install xgboost\n")

results = {}
print("="*60)
print("MODEL TRAINING")
print("="*60)

for name, model in models.items():
    print(f"Training {name}...", end=" ", flush=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"Accuracy = {acc:.4f}")

# Results table
print("\n" + "="*60)
print("FINAL ACCURACY TABLE")
print("="*60)
results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))

# Bar chart
plt.figure(figsize=(10, 5))
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_df))]
plt.barh(results_df['Model'], results_df['Accuracy'], color=colors)
plt.xlabel('Accuracy')
plt.title('ML Model Comparison — Tweet Sentiment')
plt.xlim(0.5, 1.0)
for i, v in enumerate(results_df['Accuracy']):
    plt.text(v + 0.005, i, f"{v:.4f}", va='center')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('ml_model_comparison.png', dpi=100)
plt.show()


# ------------------------------------------------------------
# 3. FEATURE IMPORTANCE (Random Forest)
# ------------------------------------------------------------
print("\n" + "="*60)
print("TOP 20 MOST IMPORTANT FEATURES (Random Forest)")
print("="*60)

rf = models["Random Forest"]
feature_names = tfidf.get_feature_names_out()
importances = rf.feature_importances_
top_idx = np.argsort(importances)[-20:][::-1]

for i in top_idx:
    print(f"  {feature_names[i]:<25} {importances[i]:.5f}")

plt.figure(figsize=(8, 8))
plt.barh([feature_names[i] for i in top_idx],
         [importances[i] for i in top_idx], color='steelblue')
plt.xlabel('Importance')
plt.title('Top 20 Important Words for Sentiment')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100)
plt.show()


# ------------------------------------------------------------
# 4. K-MEANS++ CLUSTERING
# ------------------------------------------------------------
print("\n" + "="*60)
print("K-MEANS++ CLUSTERING")
print("="*60)

svd = TruncatedSVD(n_components=50, random_state=42)
X_reduced = svd.fit_transform(X)

kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_reduced)

sil = silhouette_score(X_reduced, clusters, sample_size=5000, random_state=42)
print(f"Silhouette Score: {sil:.4f}")
print(f"Cluster sizes: {np.bincount(clusters)}")

cluster_df = pd.DataFrame({'cluster': clusters, 'sentiment': y})
print("\nCluster vs Sentiment cross-tab:")
print(pd.crosstab(cluster_df['cluster'], cluster_df['sentiment']))


# ------------------------------------------------------------
# 5. LDA TOPIC MODELING
# ------------------------------------------------------------
print("\n" + "="*60)
print("LDA TOPIC MODELING (5 topics)")
print("="*60)

cv = CountVectorizer(max_features=5000, min_df=5, max_df=0.9, stop_words='english')
X_counts = cv.fit_transform(X_text)

lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=10)
lda.fit(X_counts)

feature_names_cv = cv.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names_cv[i] for i in topic.argsort()[-10:][::-1]]
    print(f"\nTopic {topic_idx + 1}: {' '.join(top_words)}")


# ------------------------------------------------------------
# 6. BAGGING + BOOSTING
# ------------------------------------------------------------
print("\n" + "="*60)
print("BAGGING vs BOOSTING (Ensemble methods)")
print("="*60)

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=15),
    n_estimators=30, random_state=42, n_jobs=-1
)
bagging.fit(X_train, y_train)
bag_acc = accuracy_score(y_test, bagging.predict(X_test))

boosting = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=10),
    n_estimators=50, random_state=42
)
boosting.fit(X_train, y_train)
boost_acc = accuracy_score(y_test, boosting.predict(X_test))

print(f"Bagging  (30 trees): Accuracy = {bag_acc:.4f}")
print(f"Boosting (50 trees): Accuracy = {boost_acc:.4f}")


# ------------------------------------------------------------
# 7. FINAL SUMMARY
# ------------------------------------------------------------
print("\n" + "="*60)
print("FINAL SUMMARY — ALL MODELS")
print("="*60)
final = results.copy()
final['Bagging'] = bag_acc
final['Boosting'] = boost_acc
for name, acc in sorted(final.items(), key=lambda x: -x[1]):
    print(f"  {name:<25} {acc:.4f}")

print("\n✅ ML Module Complete. Plots saved.")