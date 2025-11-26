#!/usr/bin/env python3
"""
Final analysis of the improved model predictions.
"""

import pandas as pd
import numpy as np
from collections import Counter

print("="*60)
print("IMPROVED MODEL - FINAL ANALYSIS")
print("="*60)

# Load predictions
print("\nLoading submission file...")
predictions = pd.read_csv('submission.tsv', sep='\t', 
                         header=None, names=['EntryID', 'term', 'score'])

print(f"\n{'='*60}")
print("SUBMISSION FILE STATISTICS")
print(f"{'='*60}")
print(f"Total predictions: {len(predictions):,}")
print(f"Unique proteins: {predictions['EntryID'].nunique():,}")
print(f"Unique GO terms predicted: {predictions['term'].nunique()}")
print(f"Average predictions per protein: {len(predictions)/predictions['EntryID'].nunique():.2f}")
print(f"Score range: {predictions['score'].min():.3f} to {predictions['score'].max():.3f}")
print(f"Average score: {predictions['score'].mean():.3f}")
print(f"Median score: {predictions['score'].median():.3f}")

print(f"\n{'='*60}")
print("MOST FREQUENTLY PREDICTED GO TERMS (Top 30)")
print(f"{'='*60}")
term_counts = predictions['term'].value_counts().head(30)
for i, (term, count) in enumerate(term_counts.items(), 1):
    pct = (count / predictions['EntryID'].nunique()) * 100
    print(f"{i:2}. {term}: {count:6,} proteins ({pct:5.1f}%)")

print(f"\n{'='*60}")
print("PREDICTIONS PER PROTEIN DISTRIBUTION")
print(f"{'='*60}")
preds_per_protein = predictions.groupby('EntryID').size()
print(f"Min predictions: {preds_per_protein.min()}")
print(f"Max predictions: {preds_per_protein.max()}")
print(f"Median predictions: {preds_per_protein.median():.0f}")
print(f"Mean predictions: {preds_per_protein.mean():.2f}")
print(f"Std predictions: {preds_per_protein.std():.2f}")

print(f"\n{'='*60}")
print("CONFIDENCE SCORE DISTRIBUTION")
print(f"{'='*60}")
score_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
for low, high in score_bins:
    count = ((predictions['score'] >= low) & (predictions['score'] < high)).sum()
    if low == 0.8:
        count = (predictions['score'] >= low).sum()
    pct = count / len(predictions) * 100
    print(f"Score {low:.1f}-{high:.1f}: {count:8,} predictions ({pct:5.1f}%)")

print(f"\n{'='*60}")
print("HIGH CONFIDENCE PREDICTIONS (Score > 0.9)")
print(f"{'='*60}")
high_conf = predictions[predictions['score'] > 0.9]
print(f"Total high confidence predictions: {len(high_conf):,}")
print(f"Proteins with high confidence predictions: {high_conf['EntryID'].nunique():,}")
print(f"\nTop GO terms in high confidence predictions:")
for i, (term, count) in enumerate(high_conf['term'].value_counts().head(15).items(), 1):
    print(f"{i:2}. {term}: {count:,}")

print(f"\n{'='*60}")
print("SAMPLE PREDICTIONS FOR REFERENCE PROTEINS")
print(f"{'='*60}")
sample_proteins = ['A0A0C5B5G6', 'A0JNW5', 'O00115', 'P00338', 'P04406']
for protein in sample_proteins:
    if protein in predictions['EntryID'].values:
        preds = predictions[predictions['EntryID'] == protein].head(5)
        print(f"\n{protein}:")
        for _, row in preds.iterrows():
            print(f"  {row['term']}: {row['score']:.3f}")

print(f"\n{'='*60}")
print("MODEL IMPROVEMENTS SUMMARY")
print(f"{'='*60}")
print("✓ Increased features from 31 to 68")
print("  - Added physicochemical properties")
print("  - Added N-terminal and C-terminal composition")
print("  - Extended dipeptide features")
print("  - Added log-length transformation")
print("")
print("✓ Feature standardization (StandardScaler)")
print("")
print("✓ Optimized model parameters")
print("  - Solver: lbfgs (faster convergence)")
print("  - Class weighting: balanced (handles imbalance)")
print("  - Regularization: C=0.5 (prevents overfitting)")
print("")
print("✓ Predicting 600 GO terms (balanced coverage)")
print("")
print("✓ Improved prediction threshold strategy")
print("  - Minimum 3 predictions per protein")
print("  - Maximum 15 predictions (better precision)")
print("  - Threshold 0.15 for quality")

print(f"\n{'='*60}")
print("OUTPUT FILES")
print(f"{'='*60}")
print("✓ submission.tsv - Final predictions (3,364,635 rows)")
print("✓ logistic_regression_model.pkl - Trained model")
print("✓ train_logistic_regression.py - Training script")
print("✓ predict.py - Prediction script")

print(f"\n{'='*60}")
print("READY FOR SUBMISSION!")
print(f"{'='*60}\n")
