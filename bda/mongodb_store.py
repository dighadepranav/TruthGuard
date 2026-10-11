# ============================================================
# TweetSense — BDA Module 1: MongoDB Storage & Queries
# Covers BDA Labs 2, 3 (NoSQL storage + queries)
# ============================================================

import pandas as pd
import re
import os
from pymongo import MongoClient

# ------------------------------------------------------------
# 1. CONNECT TO MONGODB ATLAS
# ------------------------------------------------------------
# ⚠️ REPLACE with your actual connection string
MONGO_URI = "mongodb+srv://mini-user:pranavdighade@backend-cluster.9dyildy.mongodb.net/?appName=backend-cluster"

client = MongoClient(MONGO_URI)
db = client['tweetsense']
collection = db['tweets']

print("✅ Connected to MongoDB Atlas")
print(f"Database: {db.name}  |  Collection: {collection.name}")


# ------------------------------------------------------------
# 2. LOAD + PREPROCESS TWEETS (same pipeline)
# ------------------------------------------------------------
CSV_PATH = os.path.join('..', 'nlp', 'training.1600000.processed.noemoticon.csv')

df = pd.read_csv(
    CSV_PATH, encoding='latin-1', header=None,
    names=['sentiment', 'id', 'date', 'query', 'user', 'text']
)
df = df.sample(10000, random_state=42).reset_index(drop=True)
df['sentiment'] = df['sentiment'].map({0: 'negative', 4: 'positive'})

# Light cleaning (keep original text for storage)
def clean(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    return text.strip()

df['clean_text'] = df['text'].apply(clean)

# ------------------------------------------------------------
# 3. INSERT INTO MONGODB (batch insert for speed)
# ------------------------------------------------------------
print("\n📥 Inserting tweets into MongoDB...")
collection.delete_many({})   # clear previous data

records = df[['text', 'clean_text', 'sentiment', 'user']].to_dict('records')
collection.insert_many(records)

print(f"✅ Inserted {collection.count_documents({})} documents")


# ------------------------------------------------------------
# 4. RUN QUERIES (BDA Lab 3 — Run commands)
# ------------------------------------------------------------
print("\n" + "="*60)
print("MONGODB QUERIES")
print("="*60)

# Query 1: Count positive vs negative
print("\n[Q1] Sentiment distribution:")
pipeline = [
    {"$group": {"_id": "$sentiment", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]
for doc in collection.aggregate(pipeline):
    print(f"  {doc['_id']}: {doc['count']}")

# Query 2: Top 5 users by tweet count
print("\n[Q2] Top 5 most active users:")
pipeline = [
    {"$group": {"_id": "$user", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
    {"$limit": 5}
]
for doc in collection.aggregate(pipeline):
    print(f"  {doc['_id']}: {doc['count']} tweets")

# Query 3: Tweets containing "love"
print("\n[Q3] Sample tweets containing 'love':")
for doc in collection.find({"clean_text": {"$regex": "love", "$options": "i"}}).limit(3):
    print(f"  [{doc['sentiment']}] {doc['clean_text'][:70]}")

# Query 4: Average tweet length
print("\n[Q4] Average tweet length (characters):")
pipeline = [
    {"$project": {"len": {"$strLenCP": "$clean_text"}}},
    {"$group": {"_id": None, "avg_len": {"$avg": "$len"}}}
]
for doc in collection.aggregate(pipeline):
    print(f"  {doc['avg_len']:.1f} chars")

print("\n✅ MongoDB module complete.")