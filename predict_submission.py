#!/usr/bin/env python3
"""
Generate predictions for test set using the improved model.
Creates submission.tsv file in the required format.
"""

import pandas as pd
import numpy as np
import pickle
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

# ============== Load Model ==============
print("\n[1/4] Loading trained model...")
with open('improved_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
mlb = model_data['mlb']
scaler = model_data['scaler']
var_thresh = model_data.get('var_thresh')
selector = model_data.get('selector')
feature_names = model_data['feature_names']
top_terms = model_data['top_terms']
best_threshold = model_data.get('best_threshold', 0.5)
optimal_thresholds = model_data.get('optimal_thresholds')
config = model_data.get('config', {})

print(f"  - Model type: {config.get('model_type', 'logistic')}")
print(f"  - Number of GO terms: {len(top_terms)}")
print(f"  - Best threshold: {best_threshold:.2f}")

# ============== Feature Extraction Function ==============
def extract_advanced_features(sequence):
    """Same feature extraction as training."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    features = {}
    
    seq_len = len(sequence)
    
    if seq_len == 0:
        return {f'feat_{i}': 0.0 for i in range(150)}
    
    # 1. Amino acid composition
    aa_counts = Counter(sequence)
    for aa in amino_acids:
        features[f'aa_{aa}'] = aa_counts.get(aa, 0) / seq_len
    
    # 2. Physicochemical properties
    property_groups = {
        'hydrophobic': 'AVILMFWP',
        'polar': 'STNQCY',
        'positive': 'RKH',
        'negative': 'DE',
        'aromatic': 'FWY',
        'aliphatic': 'AILV',
        'tiny': 'AGSC',
        'small': 'ACDGNPSTV',
        'large': 'EFHIKLMQRWY',
        'proline': 'P',
    }
    for prop_name, prop_aas in property_groups.items():
        features[f'prop_{prop_name}'] = sum(aa_counts.get(aa, 0) for aa in prop_aas) / seq_len
    
    # 3. Dipeptide composition
    important_dipeptides = [
        'LL', 'AA', 'AL', 'LA', 'LE', 'EL', 'VL', 'LV', 'GL', 'LG',
        'SL', 'LS', 'KL', 'LK', 'EE', 'KK', 'RR', 'SS', 'TT', 'PP',
        'GG', 'VA', 'AV', 'IL', 'LI', 'VV', 'II', 'FF', 'YY', 'WW',
        'AG', 'GA', 'AS', 'SA', 'AT', 'TA', 'GS', 'SG', 'ST', 'TS',
        'DE', 'ED', 'KE', 'EK', 'RE', 'ER', 'RK', 'KR', 'DK', 'KD'
    ]
    
    if seq_len >= 2:
        dipeptides = [sequence[i:i+2] for i in range(seq_len - 1)]
        dp_counts = Counter(dipeptides)
        dp_total = len(dipeptides)
        for dp in important_dipeptides:
            features[f'dp_{dp}'] = dp_counts.get(dp, 0) / dp_total
    else:
        for dp in important_dipeptides:
            features[f'dp_{dp}'] = 0.0
    
    # 4. Tripeptide patterns
    important_tripeptides = [
        'LLL', 'AAA', 'GGG', 'PPP', 'SSS', 'TTT',
        'ALA', 'LAL', 'GLY', 'SER', 'THR', 'PRO',
        'LEU', 'ILE', 'VAL', 'PHE', 'TYR', 'TRP',
        'ASP', 'GLU', 'ASN', 'GLN', 'LYS', 'ARG',
        'CYS', 'MET', 'HIS', 'AGA', 'GAG', 'GGA'
    ]
    
    if seq_len >= 3:
        tripeptides = [sequence[i:i+3] for i in range(seq_len - 2)]
        tp_counts = Counter(tripeptides)
        tp_total = len(tripeptides)
        for tp in important_tripeptides:
            features[f'tp_{tp}'] = tp_counts.get(tp, 0) / tp_total
    else:
        for tp in important_tripeptides:
            features[f'tp_{tp}'] = 0.0
    
    # 5. Sequence statistics
    features['seq_length'] = seq_len
    features['seq_log_length'] = np.log1p(seq_len)
    features['seq_length_bin'] = min(seq_len // 100, 20)
    
    aa_weights = {'A': 89, 'R': 174, 'N': 132, 'D': 133, 'C': 121, 'E': 147, 'Q': 146,
                  'G': 75, 'H': 155, 'I': 131, 'L': 131, 'K': 146, 'M': 149, 'F': 165,
                  'P': 115, 'S': 105, 'T': 119, 'W': 204, 'Y': 181, 'V': 117}
    features['mol_weight'] = sum(aa_weights.get(aa, 110) * aa_counts.get(aa, 0) for aa in amino_acids) / 1000
    
    pos_charge = aa_counts.get('R', 0) + aa_counts.get('K', 0) + aa_counts.get('H', 0) * 0.1
    neg_charge = aa_counts.get('D', 0) + aa_counts.get('E', 0)
    features['charge_balance'] = (pos_charge - neg_charge) / seq_len
    features['net_charge'] = pos_charge - neg_charge
    
    entropy = 0
    for aa in amino_acids:
        p = aa_counts.get(aa, 0) / seq_len
        if p > 0:
            entropy -= p * np.log2(p)
    features['seq_entropy'] = entropy
    features['seq_complexity'] = entropy / np.log2(20)
    
    kd_scale = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'E': -3.5, 'Q': -3.5,
                'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
                'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
    features['hydrophobicity'] = sum(kd_scale.get(aa, 0) * aa_counts.get(aa, 0) for aa in sequence) / seq_len
    
    # 6. Terminal region features
    term_len = min(30, seq_len // 4) if seq_len > 10 else seq_len
    n_term = sequence[:term_len]
    c_term = sequence[-term_len:]
    
    n_counts = Counter(n_term)
    c_counts = Counter(c_term)
    
    terminal_aas = ['M', 'A', 'G', 'S', 'T', 'L', 'K', 'R']
    for aa in terminal_aas:
        features[f'nterm_{aa}'] = n_counts.get(aa, 0) / len(n_term) if n_term else 0
        features[f'cterm_{aa}'] = c_counts.get(aa, 0) / len(c_term) if c_term else 0
    
    # 7. Secondary structure propensity
    features['helix_propensity'] = sum(aa_counts.get(aa, 0) for aa in 'AELM') / seq_len
    features['sheet_propensity'] = sum(aa_counts.get(aa, 0) for aa in 'VIY') / seq_len
    features['coil_propensity'] = sum(aa_counts.get(aa, 0) for aa in 'GNPS') / seq_len
    
    return features

# ============== Load Test Sequences ==============
print("\n[2/4] Loading test sequences...")

test_sequences = {}
current_id = None
current_seq = []

with open('Test/testsuperset.fasta', 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if current_id is not None:
                test_sequences[current_id] = ''.join(current_seq)
            # Extract just the ID (before any space or pipe)
            header = line[1:]
            if '|' in header:
                match = re.search(r'\|([A-Z0-9]+)\|', header)
                if match:
                    current_id = match.group(1)
                else:
                    current_id = header.split('|')[1] if len(header.split('|')) > 1 else header.split()[0]
            else:
                current_id = header.split()[0]
            current_seq = []
        else:
            current_seq.append(line)
    if current_id is not None:
        test_sequences[current_id] = ''.join(current_seq)

print(f"  - Loaded {len(test_sequences)} test sequences")

# ============== Extract Features for Test Set ==============
print("\n[3/4] Extracting features for test sequences...")

test_features = []
test_ids = []

for i, (entry_id, sequence) in enumerate(test_sequences.items()):
    if i % 50000 == 0:
        print(f"  - Processing {i}/{len(test_sequences)}...")
    features = extract_advanced_features(sequence)
    test_features.append(features)
    test_ids.append(entry_id)

# Convert to DataFrame with same columns as training
test_df = pd.DataFrame(test_features, index=test_ids)

# Ensure same columns as training (in same order)
for col in feature_names:
    if col not in test_df.columns:
        test_df[col] = 0.0

test_df = test_df[feature_names]
test_df = test_df.fillna(0)

print(f"  - Feature matrix shape: {test_df.shape}")

# Scale features
X_test = scaler.transform(test_df.values)

# Apply feature selection if used during training
if var_thresh is not None:
    X_test = var_thresh.transform(X_test)
if selector is not None:
    X_test = selector.transform(X_test)

print(f"  - Final feature matrix shape: {X_test.shape}")

# ============== Generate Predictions ==============
print("\n[4/4] Generating predictions...")

# Get probability predictions
y_pred_proba = model.predict_proba(X_test)
print(f"  - Prediction matrix shape: {y_pred_proba.shape}")

# Generate submission with optimized thresholds
print("\n  Creating submission file...")

# Use per-class thresholds if available, otherwise use global threshold
if optimal_thresholds is not None:
    thresholds = optimal_thresholds
else:
    thresholds = np.full(len(top_terms), best_threshold)

# Calculate statistics
submission_rows = []
proteins_with_predictions = 0
total_predictions = 0

for i, entry_id in enumerate(test_ids):
    if i % 50000 == 0:
        print(f"  - Processing predictions {i}/{len(test_ids)}...")
    
    proba = y_pred_proba[i]
    
    # Get predictions above threshold
    predictions = []
    for j, (prob, term, thresh) in enumerate(zip(proba, top_terms, thresholds)):
        if prob >= thresh:
            predictions.append((term, prob))
    
    # If no predictions, use top 3 with highest probability
    if len(predictions) == 0:
        top_indices = np.argsort(proba)[-3:][::-1]
        for idx in top_indices:
            if proba[idx] > 0.1:  # Minimum confidence
                predictions.append((top_terms[idx], proba[idx]))
    
    # Limit to top 10 predictions per protein (avoid over-prediction)
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]
    
    if len(predictions) > 0:
        proteins_with_predictions += 1
        for term, prob in predictions:
            submission_rows.append({
                'Protein Id': entry_id,
                'GO Term Id': term,
                'Prediction': round(prob, 2)
            })
            total_predictions += 1

print(f"\n  - Proteins with predictions: {proteins_with_predictions}/{len(test_ids)}")
print(f"  - Total predictions: {total_predictions}")
print(f"  - Average predictions per protein: {total_predictions/len(test_ids):.2f}")

# Create submission DataFrame
submission_df = pd.DataFrame(submission_rows)

# Save submission
submission_df.to_csv('submission.tsv', sep='\t', index=False, header=False)
print(f"\n  Submission saved to 'submission.tsv'")

# Show sample
print("\n--- Sample Predictions ---")
print(submission_df.head(20).to_string(index=False))

# Statistics
print("\n--- Submission Statistics ---")
print(f"  Total rows: {len(submission_df)}")
print(f"  Unique proteins: {submission_df['Protein Id'].nunique()}")
print(f"  Unique GO terms: {submission_df['GO Term Id'].nunique()}")
print(f"  Prediction score range: {submission_df['Prediction'].min():.2f} - {submission_df['Prediction'].max():.2f}")
print(f"  Mean prediction score: {submission_df['Prediction'].mean():.2f}")

# Compare with sample submission format
print("\n--- Comparing with Sample Submission ---")
sample = pd.read_csv('sample_submission.tsv', sep='\t', header=None, names=['Protein Id', 'GO Term Id', 'Prediction'])
print(f"  Sample submission rows: {len(sample)}")
print(f"  Our submission rows: {len(submission_df)}")

print("\n" + "=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)
