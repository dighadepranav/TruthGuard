# ============================================================
# TweetSense — BDA Module 4: Data Visualization
# Covers BDA Lab 7 (Visualization using Hive/PIG/R/Tableau)
# We use Python (matplotlib) instead — same concept
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from collections import Counter

sns.set_style("whitegrid")

# ------------------------------------------------------------
# 1. LOAD TWEETS
# ------------------------------------------------------------
CSV_PATH = os.path.join('..', 'nlp', 'training.1600000.processed.noemoticon.csv')
df = pd.read_csv(CSV_PATH, encoding='latin-1', header=None,
                 names=['sentiment','id','date','query','user','text'])
df = df.sample(10000, random_state=42).reset_index(drop=True)
df['sentiment'] = df['sentiment'].map({0: 'negative', 4: 'positive'})

# ------------------------------------------------------------
# 2. TWEET LENGTH DISTRIBUTION
# ------------------------------------------------------------
df['length'] = df['text'].str.len()

plt.figure(figsize=(10, 5))
plt.hist(df[df['sentiment']=='positive']['length'], bins=50,
         alpha=0.6, label='Positive', color='green')
plt.hist(df[df['sentiment']=='negative']['length'], bins=50,
         alpha=0.6, label='Negative', color='red')
plt.xlabel('Tweet Length (characters)')
plt.ylabel('Count')
plt.title('Tweet Length Distribution by Sentiment')
plt.legend()
plt.tight_layout()
plt.savefig('viz_length_distribution.png', dpi=100)
plt.show()

# ------------------------------------------------------------
# 3. TOP 15 HASHTAGS
# ------------------------------------------------------------
def extract_hashtags(text):
    return re.findall(r'#(\w+)', text.lower())

all_hashtags = []
for t in df['text']:
    all_hashtags.extend(extract_hashtags(t))

top_hashtags = Counter(all_hashtags).most_common(15)
print("\n[Top 15 Hashtags]")
for tag, count in top_hashtags:
    print(f"  #{tag}: {count}")

if top_hashtags:
    tags, counts = zip(*top_hashtags)
    plt.figure(figsize=(10, 5))
    plt.barh(list(tags)[::-1], list(counts)[::-1], color='coral')
    plt.xlabel('Frequency')
    plt.title('Top 15 Hashtags in Tweets')
    plt.tight_layout()
    plt.savefig('viz_hashtags.png', dpi=100)
    plt.show()

# ------------------------------------------------------------
# 4. TOP 15 @MENTIONS
# ------------------------------------------------------------
def extract_mentions(text):
    return re.findall(r'@(\w+)', text.lower())

all_mentions = []
for t in df['text']:
    all_mentions.extend(extract_mentions(t))

top_mentions = Counter(all_mentions).most_common(15)
print("\n[Top 15 @Mentions]")
for user, count in top_mentions:
    print(f"  @{user}: {count}")

# ------------------------------------------------------------
# 5. SENTIMENT PIE CHART
# ------------------------------------------------------------
plt.figure(figsize=(6, 6))
counts = df['sentiment'].value_counts()
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
        colors=['#2ecc71', '#e74c3c'], startangle=90)
plt.title('Sentiment Distribution')
plt.tight_layout()
plt.savefig('viz_sentiment_pie.png', dpi=100)
plt.show()

# ------------------------------------------------------------
# 6. HOURLY ACTIVITY HEATMAP
# ------------------------------------------------------------
# Extract hour directly from string like "Mon Apr 06 22:19:45 PDT 2009"
df['hour'] = df['date'].astype(str).str.extract(r'(\d{2}):\d{2}:\d{2}')[0].astype(float)

# Drop rows where hour couldn't be parsed
df_hour = df.dropna(subset=['hour'])
df_hour = df_hour[df_hour['hour'].between(0, 23)]

if len(df_hour) > 0:
    hour_counts = df_hour['hour'].value_counts().sort_index()
    plt.figure(figsize=(12, 4))
    hour_counts.plot(kind='bar', color='steelblue')
    plt.xlabel('Hour of Day')
    plt.ylabel('Tweet Count')
    plt.title('Tweet Activity by Hour')
    plt.tight_layout()
    plt.savefig('viz_hourly_activity.png', dpi=100)
    plt.show()
    print(f"Hourly activity chart saved ({len(df_hour)} tweets)")
else:
    print("⚠️ No valid timestamps found — skipping hourly chart")

print("\n✅ Visualization module complete. 4 charts saved.")