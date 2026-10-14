# ============================================================
# TweetSense — Blockchain Module 1: Hash + Merkle Root
# Covers: Blockchain Exp 1 (Cryptography, Merkle Root Tree Hash)
# ============================================================

import hashlib
import json
import pandas as pd
import re
import os
from datetime import datetime

# ------------------------------------------------------------
# 1. HASH FUNCTION (SHA-256)
# ------------------------------------------------------------
def sha256(text):
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(text.encode()).hexdigest()

# Test
sample = "I love this product"
print("="*60)
print("SHA-256 HASH DEMO")
print("="*60)
print(f"Input : {sample}")
print(f"Hash  : {sha256(sample)}")

# ------------------------------------------------------------
# 2. MERKLE ROOT
# ------------------------------------------------------------
def merkle_root(hashes):
    """
    Compute the Merkle root of a list of hashes.
    Pairs hashes, hashes them together, repeats until one hash remains.
    """
    if not hashes:
        return None
    if len(hashes) == 1:
        return hashes[0]
    
    # If odd number, duplicate the last one
    if len(hashes) % 2 != 0:
        hashes.append(hashes[-1])
    
    # Pair adjacent hashes and hash each pair
    new_level = []
    for i in range(0, len(hashes), 2):
        combined = hashes[i] + hashes[i+1]
        new_level.append(sha256(combined))
    
    return merkle_root(new_level)


# ------------------------------------------------------------
# 3. BUILD MERKLE TREE FROM SENTIMENT RESULTS
# ------------------------------------------------------------
CSV_PATH = os.path.join('..', 'nlp', 'training.1600000.processed.noemoticon.csv')
df = pd.read_csv(CSV_PATH, encoding='latin-1', header=None,
                 names=['sentiment','id','date','query','user','text'])
df = df.sample(20, random_state=42).reset_index(drop=True)   # small batch for demo
df['sentiment'] = df['sentiment'].map({0: 'negative', 4: 'positive'})

# Create a clean result record for each tweet (simulating model output)
records = []
for _, row in df.iterrows():
    record = {
        'text': row['text'][:80],          # truncated for readability
        'sentiment': row['sentiment'],
        'timestamp': str(datetime.now())
    }
    record_json = json.dumps(record, sort_keys=True)
    record_hash = sha256(record_json)
    records.append({
        'text': record['text'],
        'sentiment': record['sentiment'],
        'hash': record_hash
    })

print("\n" + "="*60)
print("SENTIMENT RESULT HASHES")
print("="*60)
for i, r in enumerate(records[:5], 1):
    print(f"\n[{i}] Text: {r['text'][:60]}...")
    print(f"    Sentiment: {r['sentiment']}")
    print(f"    Hash: {r['hash']}")

# ------------------------------------------------------------
# 4. COMPUTE MERKLE ROOT
# ------------------------------------------------------------
leaf_hashes = [r['hash'] for r in records]
root = merkle_root(leaf_hashes.copy())

print("\n" + "="*60)
print("MERKLE ROOT")
print("="*60)
print(f"Number of leaves: {len(leaf_hashes)}")
print(f"Merkle Root: {root}")
print(f"\n💡 This single hash represents ALL {len(records)} sentiment results.")
print("   Store this one hash on blockchain → verifies all records.")

# ------------------------------------------------------------
# 5. SAVE THE ROOT FOR SMART CONTRACT
# ------------------------------------------------------------
with open('merkle_root.txt', 'w') as f:
    f.write(root)
with open('record_hashes.json', 'w') as f:
    json.dump(records, f, indent=2)

print("\n✅ Saved: merkle_root.txt, record_hashes.json")
print(f"\n🎯 Copy this root for the smart contract:")
print(f"   {root}")