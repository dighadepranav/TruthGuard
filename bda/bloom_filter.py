# ============================================================
# TweetSense — BDA Module 3: Bloom Filter
# Covers BDA Lab 6 (Bloom filter implementation)
# ============================================================

import pandas as pd
import os
import mmh3
from bitarray import bitarray

# ------------------------------------------------------------
# 1. BLOOM FILTER CLASS
# ------------------------------------------------------------
class BloomFilter:
    def __init__(self, size=100000, num_hashes=5):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)
        self.items_added = 0
    
    def _hashes(self, item):
        """Generate k hash values for an item."""
        return [mmh3.hash(item, seed=i) % self.size for i in range(self.num_hashes)]
    
    def add(self, item):
        for h in self._hashes(item):
            self.bit_array[h] = 1
        self.items_added += 1
    
    def contains(self, item):
        return all(self.bit_array[h] for h in self._hashes(item))
    
    def stats(self):
        bits_set = self.bit_array.count(1)
        fill_ratio = bits_set / self.size
        return {
            'size': self.size,
            'hashes': self.num_hashes,
            'items': self.items_added,
            'bits_set': bits_set,
            'fill_ratio': f"{fill_ratio:.2%}"
        }


# ------------------------------------------------------------
# 2. LOAD TWEETS
# ------------------------------------------------------------
CSV_PATH = os.path.join('..', 'nlp', 'training.1600000.processed.noemoticon.csv')
df = pd.read_csv(CSV_PATH, encoding='latin-1', header=None,
                 names=['sentiment','id','date','query','user','text'])
df = df.sample(10000, random_state=42).reset_index(drop=True)

# ------------------------------------------------------------
# 3. ADD TWEETS TO BLOOM FILTER
# ------------------------------------------------------------
print("="*60)
print("BLOOM FILTER — DUPLICATE TWEET DETECTION")
print("="*60)

bf = BloomFilter(size=100000, num_hashes=5)

print("\n[1] Adding 10,000 tweets to Bloom filter...")
for tweet in df['text']:
    bf.add(tweet)

stats = bf.stats()
print(f"    Filter size:    {stats['size']} bits")
print(f"    Hash functions: {stats['hashes']}")
print(f"    Items added:    {stats['items']}")
print(f"    Bits set:       {stats['bits_set']}")
print(f"    Fill ratio:     {stats['fill_ratio']}")

# ------------------------------------------------------------
# 4. CHECK FOR DUPLICATES
# ------------------------------------------------------------
print("\n[2] Checking for duplicates...")

duplicates = 0
for tweet in df['text']:
    if bf.contains(tweet):
        duplicates += 1

# Bloom filter always returns True for items it has seen — the interesting case is NEW items
print("\n[3] Testing on NEW tweets (should mostly return False):")
new_tweets = [
    "This is a completely new tweet that was never added",
    "Another brand new sentence just for testing",
    "Yet another unique tweet here",
]
for tweet in new_tweets:
    result = "MAYBE SEEN" if bf.contains(tweet) else "DEFINITELY NEW"
    print(f"    '{tweet[:40]}...' → {result}")

print("\n[4] Testing on EXISTING tweets (should return True):")
for i in [0, 100, 500]:
    tweet = df['text'].iloc[i]
    result = "MAYBE SEEN" if bf.contains(tweet) else "DEFINITELY NEW"
    print(f"    '{tweet[:40]}...' → {result}")

print("\n💡 Bloom filter use cases:")
print("   - Duplicate tweet detection (memory-efficient)")
print("   - Cache filtering (avoid DB hits)")
print("   - Spam/bot detection (known bad actors)")
print("   - No false negatives, only false positives")

print("\n✅ Bloom filter module complete.")