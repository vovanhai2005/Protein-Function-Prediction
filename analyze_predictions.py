#!/usr/bin/env python3
"""
Analyze the predictions and show statistics.
"""

import pandas as pd
from collections import Counter

print("Loading predictions...")
predictions = pd.read_csv('predictions_submission.tsv', sep='\t', 
                         header=None, names=['EntryID', 'term', 'score'])

print(f"\n=== Prediction Statistics ===")
print(f"Total predictions: {len(predictions):,}")
print(f"Unique proteins: {predictions['EntryID'].nunique():,}")
print(f"Unique GO terms predicted: {predictions['term'].nunique()}")
print(f"Average predictions per protein: {len(predictions)/predictions['EntryID'].nunique():.2f}")
print(f"Score range: {predictions['score'].min():.3f} to {predictions['score'].max():.3f}")
print(f"Average score: {predictions['score'].mean():.3f}")

print(f"\n=== Most Frequently Predicted GO Terms ===")
term_counts = predictions['term'].value_counts().head(20)
for term, count in term_counts.items():
    pct = (count / predictions['EntryID'].nunique()) * 100
    print(f"{term}: {count:,} proteins ({pct:.1f}%)")

print(f"\n=== Predictions per Protein Distribution ===")
preds_per_protein = predictions.groupby('EntryID').size()
print(f"Min: {preds_per_protein.min()}")
print(f"Max: {preds_per_protein.max()}")
print(f"Median: {preds_per_protein.median():.0f}")
print(f"Mean: {preds_per_protein.mean():.2f}")

print(f"\n=== Sample High-Confidence Predictions ===")
high_conf = predictions[predictions['score'] > 0.5].head(20)
print(high_conf.to_string(index=False))

print(f"\n=== Files Created ===")
print(f"✓ logistic_regression_model.pkl - Trained model")
print(f"✓ predictions_submission.tsv - Test predictions ({len(predictions):,} rows)")
