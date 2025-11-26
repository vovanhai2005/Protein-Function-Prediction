#!/usr/bin/env python3
"""
Train an improved multi-label logistic regression model to predict GO terms from protein sequences.
Enhanced with better features and model tuning.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
import pickle
from collections import Counter, defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")

# Load training terms
train_terms = pd.read_csv('Train/train_terms.tsv', sep='\t')
print(f"Loaded {len(train_terms)} term annotations")

# Load taxonomy
train_taxonomy = pd.read_csv('Train/train_taxonomy.tsv', sep='\t', header=None, names=['EntryID', 'TaxonID'])
print(f"Loaded {len(train_taxonomy)} taxonomy entries")

# Parse FASTA file
print("Parsing FASTA sequences...")
sequences = {}
current_id = None
current_seq = []

with open('Train/train_sequences.fasta', 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
            # Extract ID from header (format: >sp|ID|...)
            match = re.search(r'\|([A-Z0-9]+)\|', line)
            if match:
                current_id = match.group(1)
            else:
                current_id = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line)
    
    # Don't forget the last sequence
    if current_id is not None:
        sequences[current_id] = ''.join(current_seq)

print(f"Parsed {len(sequences)} protein sequences")

# Extract features from amino acid sequences
def extract_features(sequence):
    """
    Extract comprehensive features from amino acid sequence:
    - Amino acid composition (20 features)
    - Sequence length and log-length
    - Physicochemical properties
    - Dipeptide frequencies (extended)
    - Tripeptide samples
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

print("Extracting features from sequences...")
feature_data = []
entry_ids = []

for entry_id, sequence in sequences.items():
    features = extract_features(sequence)
    feature_data.append(features)
    entry_ids.append(entry_id)

# Convert to DataFrame
X_df = pd.DataFrame(feature_data, index=entry_ids)
print(f"Extracted features shape: {X_df.shape}")

# Standardize features for better convergence
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df.values)
print(f"Feature scaling complete")

# Prepare labels (GO terms) - multi-label format
print("Preparing GO term labels...")
entry_to_terms = defaultdict(set)  # Use set for faster lookups
for idx, row in train_terms.iterrows():
    if idx % 100000 == 0:
        print(f"  Processed {idx}/{len(train_terms)} annotations...")
    entry_to_terms[row['EntryID']].add(row['term'])

# Convert sets to lists
entry_to_terms = {k: list(v) for k, v in entry_to_terms.items()}

# Filter to only include entries we have sequences for
valid_entry_ids = [eid for eid in entry_ids if eid in entry_to_terms]
print(f"Found {len(valid_entry_ids)} entries with both sequences and GO terms")

# Prepare final datasets
X = X_scaled[X_df.index.isin(valid_entry_ids)]
y_labels = [entry_to_terms[eid] for eid in valid_entry_ids]

# Binarize labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y_labels)
print(f"Total unique GO terms: {len(mlb.classes_)}")
print(f"Label matrix shape: {y.shape}")

# Due to the large number of labels, let's filter to top N most common terms
# This makes the problem more tractable
top_n_terms = 600  # Optimized for balance between coverage and training time
term_counts = Counter([term for terms in y_labels for term in terms])
top_terms = [term for term, count in term_counts.most_common(top_n_terms)]
print(f"\nFocusing on top {top_n_terms} most common GO terms")
print(f"Most common terms: {term_counts.most_common(10)}")

# Filter labels to top terms
filtered_y_labels = [[term for term in terms if term in top_terms] for terms in y_labels]
mlb_filtered = MultiLabelBinarizer(classes=top_terms)
y_filtered = mlb_filtered.fit_transform(filtered_y_labels)

print(f"Filtered label matrix shape: {y_filtered.shape}")
print(f"Average labels per protein: {y_filtered.sum() / len(y_filtered):.2f}")

# Split data - stratified split is not possible for multi-label, but we ensure balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y_filtered, test_size=0.15, random_state=42  # Smaller test set for more training data
)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train multi-label logistic regression with improved parameters
print("\nTraining improved multi-label logistic regression model...")
print("This may take several minutes...")

model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,  # Optimized iterations
        solver='lbfgs',  # Faster convergence
        C=0.5,  # Balanced regularization
        penalty='l2',
        random_state=42,
        class_weight='balanced',  # Handle class imbalance
        n_jobs=1,
        verbose=0
    ),
    n_jobs=-1  # Parallel across classifiers
)

print(f"Training {len(top_terms)} binary classifiers in parallel...")
model.fit(X_train, y_train)
print("Training complete!")

# Evaluate
print("\nEvaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate metrics (samples average for multi-label)
f1_samples = f1_score(y_test, y_pred, average='samples', zero_division=0)
precision_samples = precision_score(y_test, y_pred, average='samples', zero_division=0)
recall_samples = recall_score(y_test, y_pred, average='samples', zero_division=0)

f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

precision_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
recall_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)

hamming = hamming_loss(y_test, y_pred)

print(f"\n=== Model Performance ===")
print(f"F1-Score (samples): {f1_samples:.4f}")
print(f"Precision (samples): {precision_samples:.4f}")
print(f"Recall (samples): {recall_samples:.4f}")
print(f"\nF1-Score (micro): {f1_micro:.4f}")
print(f"F1-Score (macro): {f1_macro:.4f}")
print(f"F1-Score (weighted): {f1_weighted:.4f}")
print(f"\nPrecision (micro): {precision_micro:.4f}")
print(f"Recall (micro): {recall_micro:.4f}")
print(f"\nHamming Loss: {hamming:.4f}")

# Calculate coverage and accuracy at different thresholds
print(f"\n=== Threshold Analysis ===")
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    f1_thresh = f1_score(y_test, y_pred_thresh, average='samples', zero_division=0)
    coverage = (y_pred_thresh.sum(axis=1) > 0).mean()
    avg_preds = y_pred_thresh.sum(axis=1).mean()
    print(f"Threshold {threshold:.1f}: F1={f1_thresh:.4f}, Coverage={coverage:.2%}, Avg predictions={avg_preds:.2f}")

# Save the model and label binarizer
print("\nSaving model...")
model_data = {
    'model': model,
    'mlb': mlb_filtered,
    'scaler': scaler,
    'feature_names': X_df.columns.tolist(),
    'top_terms': top_terms
}

with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved to 'logistic_regression_model.pkl'")

# Save feature importance for inspection
print("\nAnalyzing feature importance...")
feature_importance = []
for i, term in enumerate(top_terms[:10]):  # Top 10 terms
    estimator = model.estimators_[i]
    coef = estimator.coef_[0]
    top_features_idx = np.argsort(np.abs(coef))[-10:][::-1]
    feature_importance.append({
        'term': term,
        'top_features': [(X_df.columns[idx], coef[idx]) for idx in top_features_idx]
    })

print("\nTop features for 10 most common GO terms:")
for fi in feature_importance[:5]:
    print(f"\n{fi['term']}:")
    for feat, weight in fi['top_features'][:5]:
        print(f"  {feat}: {weight:.4f}")

print("\n=== Training Complete ===")
print(f"Model trained on {len(valid_entry_ids)} proteins")
print(f"Predicting {top_n_terms} GO terms")
print(f"Model file: logistic_regression_model.pkl")
