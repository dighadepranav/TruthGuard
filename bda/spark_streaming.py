# ============================================================
# TweetSense — BDA Module 5: Spark Streaming Simulation
# Real-time micro-batch processing of tweet streams
# (Spark-compatible logic, no worker dependency)
# ============================================================

import os
import re
import time
from collections import Counter, deque

STREAM_DIR = "stream_input"
os.makedirs(STREAM_DIR, exist_ok=True)

POSITIVE_WORDS = {'love','great','amazing','good','happy','excellent',
                  'awesome','best','thank','nice','wonderful','fantastic',
                  'brilliant','enjoy'}
NEGATIVE_WORDS = {'hate','terrible','worst','bad','sad','awful','horrible',
                  'suck','sorry','poor','disappointing','boring','waste','angry'}


def process_batch(batch_num, lines):
    """Simulates one Spark micro-batch on a set of lines."""
    if not lines:
        print(f"[Batch {batch_num}] Empty batch — no new data")
        return

    # Clean
    cleaned = [re.sub(r'[^a-z\s]', ' ', ln.lower()) for ln in lines]

    # Tokenize
    words = []
    for c in cleaned:
        words.extend(c.split())

    # Filter (length > 3)
    filtered = [w for w in words if len(w) > 3]

    # Word count (reduceByKey equivalent)
    counter = Counter(filtered)
    top10 = counter.most_common(10)

    # Sentiment analysis
    pos = sum(counter[w] for w in counter if w in POSITIVE_WORDS)
    neg = sum(counter[w] for w in counter if w in NEGATIVE_WORDS)
    total = len(filtered)

    sentiment = ("POSITIVE 😊" if pos > neg else
                 "NEGATIVE 😞" if neg > pos else "NEUTRAL 😐")

    # Print formatted output (identical to Spark DStream format)
    print("\n" + "="*60)
    print(f"[BATCH {batch_num} PROCESSED] Total words: {total}")
    print(f"  Positive words: {pos}  |  Negative words: {neg}")
    print(f"  Batch sentiment: {sentiment}")
    print(f"\n  Top 10 words:")
    for word, count in top10:
        print(f"    {word:<15} {count}")
    print("="*60)


def main():
    print("="*60)
    print("SPARK STREAMING STARTED (micro-batch mode)")
    print(f"Monitoring: {os.path.abspath(STREAM_DIR)}")
    print("Batch interval: 10 seconds")
    print("Press Ctrl+C to stop.")
    print("="*60)

    processed = set()
    batch_num = 0
    no_data_ticks = 0

    while True:
        files = sorted([f for f in os.listdir(STREAM_DIR)
                        if f.endswith('.txt') and f not in processed])

        if files:
            batch_num += 1
            all_lines = []
            for f in files:
                path = os.path.join(STREAM_DIR, f)
                with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                    all_lines.extend([ln.strip() for ln in fh if ln.strip()])
                processed.add(f)
            process_batch(batch_num, all_lines)
            no_data_ticks = 0
        else:
            no_data_ticks += 1
            if no_data_ticks % 6 == 0:   # don't spam — every 60 sec
                print("[Batch] No new data this interval.")

        time.sleep(10)   # 10-second micro-batch interval


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[STOPPED] Streaming terminated by user.")