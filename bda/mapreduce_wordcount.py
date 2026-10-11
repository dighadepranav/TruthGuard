# ============================================================
# TweetSense — BDA Module 2: MapReduce Word Count
# Covers BDA Lab 4 (Word count using MapReduce)
# ============================================================

import pandas as pd
import re
import os
import nltk
from collections import defaultdict

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ------------------------------------------------------------
# 1. LOAD TWEETS
# ------------------------------------------------------------
CSV_PATH = os.path.join('..', 'nlp', 'training.1600000.processed.noemoticon.csv')
df = pd.read_csv(CSV_PATH, encoding='latin-1', header=None,
                 names=['sentiment','id','date','query','user','text'])
df = df.sample(10000, random_state=42).reset_index(drop=True)

stop_words = set(stopwords.words('english'))


# ------------------------------------------------------------
# 2. MAP FUNCTION
#    Input:  one line of text
#    Output: list of (word, 1) pairs
# ------------------------------------------------------------
def mapper(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    return [(w, 1) for w in tokens if w not in stop_words and len(w) > 2]


# ------------------------------------------------------------
# 3. SHUFFLE + SORT
#    Group all values by key
# ------------------------------------------------------------
def shuffle(mapped_pairs):
    grouped = defaultdict(list)
    for key, value in mapped_pairs:
        grouped[key].append(value)
    return grouped


# ------------------------------------------------------------
# 4. REDUCE FUNCTION
#    Input:  key, list of values
#    Output: key, sum of values
# ------------------------------------------------------------
def reducer(key, values):
    return (key, sum(values))


# ------------------------------------------------------------
# 5. RUN MAPREDUCE
# ------------------------------------------------------------
print("="*60)
print("MAPREDUCE WORD COUNT")
print("="*60)

# MAP
print("\n[1] MAP phase — mapping 10,000 tweets...")
mapped = []
for text in df['text']:
    mapped.extend(mapper(text))
print(f"    Emitted {len(mapped)} key-value pairs")

# SHUFFLE
print("\n[2] SHUFFLE phase — grouping by key...")
grouped = shuffle(mapped)
print(f"    Grouped into {len(grouped)} unique words")

# REDUCE
print("\n[3] REDUCE phase — summing counts...")
reduced = [reducer(k, v) for k, v in grouped.items()]
reduced.sort(key=lambda x: -x[1])

# ------------------------------------------------------------
# 6. TOP 20 WORDS
# ------------------------------------------------------------
print("\n" + "="*60)
print("TOP 20 MOST FREQUENT WORDS")
print("="*60)
for word, count in reduced[:20]:
    bar = '█' * min(int(count / 100), 40)
    print(f"  {word:<15} {count:>6}  {bar}")

print("\n✅ MapReduce module complete.")