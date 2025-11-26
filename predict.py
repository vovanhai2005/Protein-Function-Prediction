#!/usr/bin/env python3
"""
Generate predictions for test set using the trained logistic regression model.
"""

import pandas as pd
import numpy as np
import pickle
from collections import Counter
import re

print("Loading trained model...")
with open('logistic_regression_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
mlb = model_data['mlb']
scaler = model_data['scaler']
feature_names = model_data['feature_names']
top_terms = model_data['top_terms']

print(f"Model loaded - predicts {len(top_terms)} GO terms")

# Parse test FASTA file
print("Parsing test sequences...")
sequences = {}
current_id = None
current_seq = []

with open('Test/testsuperset.fasta', 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
            # Extract ID from header (format: >ID taxon)
            parts = line[1:].split()
            current_id = parts[0]
            current_seq = []
        else:
            current_seq.append(line)
    
    # Don't forget the last sequence
    if current_id is not None:
        sequences[current_id] = ''.join(current_seq)

print(f"Parsed {len(sequences)} test sequences")

# Extract features from amino acid sequences (same as training)
def extract_features(sequence):
    """
    Extract comprehensive features from amino acid sequence:
    - Amino acid composition (20 features)
    - Sequence length and log-length
    - Physicochemical properties
    - Dipeptide frequencies (extended)
    - N-terminal and C-terminal composition
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    features = {}
    
    # Basic properties
    seq_len = len(sequence)
    features['length'] = seq_len
    features['log_length'] = np.log1p(seq_len)
    
    if seq_len == 0:
        # Return zero features for empty sequences
        for aa in amino_acids:
            features[f'aa_{aa}'] = 0.0
        features['length'] = 0
        features['log_length'] = 0
        
        # Physicochemical properties
        features['hydrophobic_ratio'] = 0.0
        features['polar_ratio'] = 0.0
        features['charged_ratio'] = 0.0
        features['positive_ratio'] = 0.0
        features['negative_ratio'] = 0.0
        features['aromatic_ratio'] = 0.0
        features['aliphatic_ratio'] = 0.0
        features['tiny_ratio'] = 0.0
        features['small_ratio'] = 0.0
        features['large_ratio'] = 0.0
        
        # Add dipeptide features
        common_dipeptides = ['AA', 'AL', 'LL', 'LA', 'GG', 'VA', 'AV', 'EE', 'KK', 'RR',
                            'SS', 'TT', 'PP', 'DD', 'NN', 'QQ', 'FF', 'WW', 'YY', 'CC']
        for dp in common_dipeptides:
            features[f'di_{dp}'] = 0.0
        
        # Terminal composition
        for aa in ['A', 'M', 'G', 'S', 'T', 'L', 'K', 'R']:
            features[f'n_term_{aa}'] = 0.0
            features[f'c_term_{aa}'] = 0.0
            
        return features
    
    # Amino acid composition (normalized)
    aa_counts = Counter(sequence)
    for aa in amino_acids:
        features[f'aa_{aa}'] = aa_counts.get(aa, 0) / seq_len
    
    # Physicochemical properties
    hydrophobic = 'AVILMFWP'
    polar = 'STNQCY'
    positive = 'RK'
    negative = 'DE'
    aromatic = 'FWY'
    aliphatic = 'AILV'
    tiny = 'AGSCT'
    small = 'ABCDGNPSTV'
    large = 'EFHIKLMQRWY'
    
    features['hydrophobic_ratio'] = sum(aa_counts.get(aa, 0) for aa in hydrophobic) / seq_len
    features['polar_ratio'] = sum(aa_counts.get(aa, 0) for aa in polar) / seq_len
    features['charged_ratio'] = sum(aa_counts.get(aa, 0) for aa in positive + negative) / seq_len
    features['positive_ratio'] = sum(aa_counts.get(aa, 0) for aa in positive) / seq_len
    features['negative_ratio'] = sum(aa_counts.get(aa, 0) for aa in negative) / seq_len
    features['aromatic_ratio'] = sum(aa_counts.get(aa, 0) for aa in aromatic) / seq_len
    features['aliphatic_ratio'] = sum(aa_counts.get(aa, 0) for aa in aliphatic) / seq_len
    features['tiny_ratio'] = sum(aa_counts.get(aa, 0) for aa in tiny) / seq_len
    features['small_ratio'] = sum(aa_counts.get(aa, 0) for aa in small) / seq_len
    features['large_ratio'] = sum(aa_counts.get(aa, 0) for aa in large) / seq_len
    
    # Dipeptide frequencies (extended set)
    dipeptides = [sequence[i:i+2] for i in range(len(sequence)-1)]
    dipeptide_counts = Counter(dipeptides)
    
    # Top common dipeptides
    common_dipeptides = ['AA', 'AL', 'LL', 'LA', 'GG', 'VA', 'AV', 'EE', 'KK', 'RR',
                        'SS', 'TT', 'PP', 'DD', 'NN', 'QQ', 'FF', 'WW', 'YY', 'CC']
    for dp in common_dipeptides:
        features[f'di_{dp}'] = dipeptide_counts.get(dp, 0) / max(len(dipeptides), 1)
    
    # N-terminal and C-terminal composition (first/last 10 residues or 10%)
    terminal_len = max(10, int(seq_len * 0.1))
    n_terminal = sequence[:terminal_len]
    c_terminal = sequence[-terminal_len:]
    
    n_term_counts = Counter(n_terminal)
    c_term_counts = Counter(c_terminal)
    
    # Key amino acids in terminals
    for aa in ['A', 'M', 'G', 'S', 'T', 'L', 'K', 'R']:
        features[f'n_term_{aa}'] = n_term_counts.get(aa, 0) / len(n_terminal)
        features[f'c_term_{aa}'] = c_term_counts.get(aa, 0) / len(c_terminal)
    
    return features

print("Extracting features from test sequences...")
feature_data = []
entry_ids = []

for i, (entry_id, sequence) in enumerate(sequences.items()):
    if i % 10000 == 0:
        print(f"  Processed {i}/{len(sequences)} sequences...")
    features = extract_features(sequence)
    feature_data.append(features)
    entry_ids.append(entry_id)

# Convert to DataFrame
X_test_df = pd.DataFrame(feature_data, index=entry_ids)
print(f"Extracted features shape: {X_test_df.shape}")

# Ensure features are in the same order as training and scale
X_test = X_test_df[feature_names].values
X_test_scaled = scaler.transform(X_test)

# Make predictions (probabilities)
print("Generating predictions...")
y_pred_proba = model.predict_proba(X_test_scaled)

print(f"Predictions shape: {y_pred_proba.shape}")

# Generate submission file
print("Creating submission file...")
submission_rows = []

for i, entry_id in enumerate(entry_ids):
    if i % 10000 == 0:
        print(f"  Processed {i}/{len(entry_ids)} entries...")
    
    # Get probabilities for this entry
    probabilities = y_pred_proba[i]
    
    # Get top predicted GO terms (sorted by probability)
    top_indices = np.argsort(probabilities)[::-1]
    
    # Add predictions for GO terms with probability > threshold
    # Or at least top N predictions
    threshold = 0.15  # Slightly higher threshold for better precision
    min_predictions = 3  # At least 3 predictions
    
    added = 0
    for idx in top_indices:
        prob = probabilities[idx]
        if prob > threshold or added < min_predictions:
            go_term = top_terms[idx]
            submission_rows.append({
                'EntryID': entry_id,
                'term': go_term,
                'score': round(prob, 3)
            })
            added += 1
        
        # Limit to top 15 predictions per protein for better precision
        if added >= 15:
            break

# Create DataFrame
submission_df = pd.DataFrame(submission_rows)

# Save to file
output_file = 'submission.tsv'
submission_df.to_csv(output_file, sep='\t', index=False, header=False)

print(f"\n=== Prediction Complete ===")
print(f"Total entries: {len(entry_ids)}")
print(f"Total predictions: {len(submission_rows)}")
print(f"Average predictions per protein: {len(submission_rows)/len(entry_ids):.2f}")
print(f"Output file: {output_file}")

# Show sample predictions
print("\nSample predictions:")
print(submission_df.head(20))
