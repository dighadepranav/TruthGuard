# ============================================================
# Tweet Feeder — Simulates real-time tweet stream
# Reads batches from Sentiment140 and drops them as files
# ============================================================

import time
import os
import pandas as pd

STREAM_DIR = "stream_input"
os.makedirs(STREAM_DIR, exist_ok=True)

# Load tweets
print("Loading tweets...")
df = pd.read_csv(
    '../nlp/training.1600000.processed.noemoticon.csv',
    encoding='latin-1', header=None,
    names=['sentiment', 'id', 'date', 'query', 'user', 'text']
)
df = df.sample(50, random_state=42).reset_index(drop=True)
print(f"Loaded {len(df)} tweets. Starting stream feed...\n")

# Send 5 batches of 10 tweets each
batch_size = 10
for batch_num in range(5):
    start = batch_num * batch_size
    end = start + batch_size
    batch = df.iloc[start:end]
    
    filename = os.path.join(STREAM_DIR, f"batch_{batch_num+1}_{int(time.time())}.txt")
    with open(filename, 'w', encoding='utf-8') as f:
        for text in batch['text']:
            f.write(text + '\n')
    
    print(f"✅ Sent batch {batch_num+1} ({len(batch)} tweets) → {filename}")
    time.sleep(10)   # wait 10 seconds between batches

print("\n✅ All batches sent. Stream complete.")