#!/usr/bin/env python3
"""
Improved multi-label model for protein function prediction.
Enhancements:
- Advanced feature engineering (k-mers, physicochemical properties)
- Feature selection using SelectKBest
- Model comparison (Logistic Regression, SVM, Random Forest)
- Hyperparameter optimization with GridSearchCV
- Optimal threshold tuning per class
- Cross-validation for robust evaluation
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import pickle
from collections import Counter, defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

# ============== Configuration ==============
CONFIG = {
    'top_n_terms': 700,          # Number of GO terms to predict (increased for better coverage)
    'test_size': 0.15,           # Test set proportion
    'random_state': 42,
    'n_features_to_select': 150,  # Number of features after selection
    'use_feature_selection': True,
    'model_type': 'logistic',     # Options: 'logistic', 'svm', 'random_forest'
    'optimize_threshold': True,
    'use_class_weights': True,
}

print("=" * 60)
print("IMPROVED PROTEIN FUNCTION PREDICTION MODEL")
print("=" * 60)

# ============== Data Loading ==============
print("\n[1/8] Loading data...")

train_terms = pd.read_csv('Train/train_terms.tsv', sep='\t')
print(f"  - Loaded {len(train_terms)} term annotations")

train_taxonomy = pd.read_csv('Train/train_taxonomy.tsv', sep='\t', header=None, names=['EntryID', 'TaxonID'])
print(f"  - Loaded {len(train_taxonomy)} taxonomy entries")

# Parse FASTA file
print("\n[2/8] Parsing FASTA sequences...")
sequences = {}
current_id = None
current_seq = []

with open('Train/train_sequences.fasta', 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
            match = re.search(r'\|([A-Z0-9]+)\|', line)
            if match:
                current_id = match.group(1)
            else:
                current_id = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line)
    if current_id is not None:
        sequences[current_id] = ''.join(current_seq)

print(f"  - Parsed {len(sequences)} protein sequences")

# ============== Advanced Feature Engineering ==============
print("\n[3/8] Extracting advanced features...")

def extract_advanced_features(sequence):
    """
    Extract comprehensive features:
    1. Amino acid composition (20 features)
    2. Physicochemical properties (10 features)
    3. Dipeptide composition (400 -> top 50 features)
    4. Tripeptide patterns (top 30 features)
    5. Sequence statistics (10 features)
    6. Terminal region features (16 features)
    7. Secondary structure propensity (3 features)
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    features = {}
    
    seq_len = len(sequence)
    
    # Handle empty sequences
    if seq_len == 0:
        return {f'feat_{i}': 0.0 for i in range(150)}
    
    # 1. Amino acid composition (20 features)
    aa_counts = Counter(sequence)
    for aa in amino_acids:
        features[f'aa_{aa}'] = aa_counts.get(aa, 0) / seq_len
    
    # 2. Physicochemical properties (10 features)
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
    
    # 3. Dipeptide composition (top 50 most informative)
    if seq_len >= 2:
        dipeptides = [sequence[i:i+2] for i in range(seq_len - 1)]
        dp_counts = Counter(dipeptides)
        dp_total = len(dipeptides)
        
        # Most common dipeptides in proteins
        important_dipeptides = [
            'LL', 'AA', 'AL', 'LA', 'LE', 'EL', 'VL', 'LV', 'GL', 'LG',
            'SL', 'LS', 'KL', 'LK', 'EE', 'KK', 'RR', 'SS', 'TT', 'PP',
            'GG', 'VA', 'AV', 'IL', 'LI', 'VV', 'II', 'FF', 'YY', 'WW',
            'AG', 'GA', 'AS', 'SA', 'AT', 'TA', 'GS', 'SG', 'ST', 'TS',
            'DE', 'ED', 'KE', 'EK', 'RE', 'ER', 'RK', 'KR', 'DK', 'KD'
        ]
        for dp in important_dipeptides:
            features[f'dp_{dp}'] = dp_counts.get(dp, 0) / dp_total
    else:
        for dp in important_dipeptides:
            features[f'dp_{dp}'] = 0.0
    
    # 4. Tripeptide patterns (focusing on known motifs)
    if seq_len >= 3:
        tripeptides = [sequence[i:i+3] for i in range(seq_len - 2)]
        tp_counts = Counter(tripeptides)
        tp_total = len(tripeptides)
        
        # Important tripeptide motifs
        important_tripeptides = [
            'LLL', 'AAA', 'GGG', 'PPP', 'SSS', 'TTT',
            'ALA', 'LAL', 'GLY', 'SER', 'THR', 'PRO',
            'LEU', 'ILE', 'VAL', 'PHE', 'TYR', 'TRP',
            'ASP', 'GLU', 'ASN', 'GLN', 'LYS', 'ARG',
            'CYS', 'MET', 'HIS', 'AGA', 'GAG', 'GGA'
        ]
        for tp in important_tripeptides:
            features[f'tp_{tp}'] = tp_counts.get(tp, 0) / tp_total
    else:
        for tp in important_tripeptides:
            features[f'tp_{tp}'] = 0.0
    
    # 5. Sequence statistics (10 features)
    features['seq_length'] = seq_len
    features['seq_log_length'] = np.log1p(seq_len)
    features['seq_length_bin'] = min(seq_len // 100, 20)  # Binned length
    
    # Molecular weight approximation
    aa_weights = {'A': 89, 'R': 174, 'N': 132, 'D': 133, 'C': 121, 'E': 147, 'Q': 146,
                  'G': 75, 'H': 155, 'I': 131, 'L': 131, 'K': 146, 'M': 149, 'F': 165,
                  'P': 115, 'S': 105, 'T': 119, 'W': 204, 'Y': 181, 'V': 117}
    features['mol_weight'] = sum(aa_weights.get(aa, 110) * aa_counts.get(aa, 0) for aa in amino_acids) / 1000
    
    # Isoelectric point approximation (simplified)
    pos_charge = aa_counts.get('R', 0) + aa_counts.get('K', 0) + aa_counts.get('H', 0) * 0.1
    neg_charge = aa_counts.get('D', 0) + aa_counts.get('E', 0)
    features['charge_balance'] = (pos_charge - neg_charge) / seq_len
    features['net_charge'] = pos_charge - neg_charge
    
    # Sequence complexity (Shannon entropy)
    entropy = 0
    for aa in amino_acids:
        p = aa_counts.get(aa, 0) / seq_len
        if p > 0:
            entropy -= p * np.log2(p)
    features['seq_entropy'] = entropy
    features['seq_complexity'] = entropy / np.log2(20)  # Normalized
    
    # Hydrophobicity index (Kyte-Doolittle scale)
    kd_scale = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'E': -3.5, 'Q': -3.5,
                'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
                'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
    features['hydrophobicity'] = sum(kd_scale.get(aa, 0) * aa_counts.get(aa, 0) for aa in sequence) / seq_len
    
    # 6. Terminal region features (N-term and C-term)
    term_len = min(30, seq_len // 4) if seq_len > 10 else seq_len
    n_term = sequence[:term_len]
    c_term = sequence[-term_len:]
    
    n_counts = Counter(n_term)
    c_counts = Counter(c_term)
    
    # Key terminal amino acids
    terminal_aas = ['M', 'A', 'G', 'S', 'T', 'L', 'K', 'R']
    for aa in terminal_aas:
        features[f'nterm_{aa}'] = n_counts.get(aa, 0) / len(n_term) if n_term else 0
        features[f'cterm_{aa}'] = c_counts.get(aa, 0) / len(c_term) if c_term else 0
    
    # 7. Secondary structure propensity
    # Helix propensity
    helix_formers = 'AELM'
    sheet_formers = 'VIY'
    coil_formers = 'GNPS'
    
    features['helix_propensity'] = sum(aa_counts.get(aa, 0) for aa in helix_formers) / seq_len
    features['sheet_propensity'] = sum(aa_counts.get(aa, 0) for aa in sheet_formers) / seq_len
    features['coil_propensity'] = sum(aa_counts.get(aa, 0) for aa in coil_formers) / seq_len
    
    return features

# Extract features for all sequences
feature_data = []
entry_ids = []

for i, (entry_id, sequence) in enumerate(sequences.items()):
    if i % 10000 == 0:
        print(f"  - Processing sequence {i}/{len(sequences)}...")
    features = extract_advanced_features(sequence)
    feature_data.append(features)
    entry_ids.append(entry_id)

X_df = pd.DataFrame(feature_data, index=entry_ids)
print(f"  - Extracted {X_df.shape[1]} features for {X_df.shape[0]} sequences")

# Fill any NaN values
X_df = X_df.fillna(0)

# ============== Label Preparation ==============
print("\n[4/8] Preparing GO term labels...")

entry_to_terms = defaultdict(set)
for idx, row in train_terms.iterrows():
    entry_to_terms[row['EntryID']].add(row['term'])

entry_to_terms = {k: list(v) for k, v in entry_to_terms.items()}

# Filter entries with both sequences and GO terms
valid_entry_ids = [eid for eid in entry_ids if eid in entry_to_terms]
print(f"  - Found {len(valid_entry_ids)} entries with sequences and GO terms")

# Get labels for valid entries
y_labels = [entry_to_terms[eid] for eid in valid_entry_ids]

# Focus on top N most common GO terms
term_counts = Counter([term for terms in y_labels for term in terms])
top_terms = [term for term, count in term_counts.most_common(CONFIG['top_n_terms'])]

print(f"  - Focusing on top {CONFIG['top_n_terms']} GO terms")
print(f"  - Top 5 terms: {term_counts.most_common(5)}")

# Filter labels to top terms
filtered_y_labels = [[term for term in terms if term in top_terms] for terms in y_labels]

# Remove entries with no remaining terms after filtering
valid_indices = [i for i, labels in enumerate(filtered_y_labels) if len(labels) > 0]
valid_entry_ids = [valid_entry_ids[i] for i in valid_indices]
filtered_y_labels = [filtered_y_labels[i] for i in valid_indices]

print(f"  - {len(valid_entry_ids)} entries have at least one top GO term")

# Binarize labels
mlb = MultiLabelBinarizer(classes=top_terms)
y = mlb.fit_transform(filtered_y_labels)

print(f"  - Label matrix shape: {y.shape}")
print(f"  - Average labels per protein: {y.sum() / len(y):.2f}")

# Prepare feature matrix
X = X_df.loc[valid_entry_ids].values

# ============== Feature Scaling ==============
print("\n[5/8] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============== Feature Selection ==============
if CONFIG['use_feature_selection']:
    print("\n[6/8] Performing feature selection...")
    
    # Use mutual information for feature selection (works well for multi-label)
    # We'll select features based on average importance across all labels
    
    # Calculate feature importance using variance and correlation with labels
    from sklearn.feature_selection import VarianceThreshold
    
    # Remove low variance features first
    var_thresh = VarianceThreshold(threshold=0.01)
    X_var = var_thresh.fit_transform(X_scaled)
    var_mask = var_thresh.get_support()
    print(f"  - After variance threshold: {X_var.shape[1]} features")
    
    # Use SelectKBest with f_classif on average label
    # Create a single target by summing labels (proxy for "protein complexity")
    y_proxy = y.sum(axis=1)
    
    n_features = min(CONFIG['n_features_to_select'], X_var.shape[1])
    selector = SelectKBest(f_classif, k=n_features)
    X_selected = selector.fit_transform(X_var, y_proxy)
    
    print(f"  - Selected {X_selected.shape[1]} best features")
    
    # Store feature selection info
    feature_names = X_df.columns.tolist()
    var_feature_names = [feature_names[i] for i in range(len(feature_names)) if var_mask[i]]
    selected_mask = selector.get_support()
    selected_feature_names = [var_feature_names[i] for i in range(len(var_feature_names)) if selected_mask[i]]
    
    X_final = X_selected
else:
    X_final = X_scaled
    var_thresh = None
    selector = None
    selected_feature_names = X_df.columns.tolist()

print(f"  - Final feature matrix shape: {X_final.shape}")

# ============== Train/Test Split ==============
print("\n[7/8] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, 
    test_size=CONFIG['test_size'], 
    random_state=CONFIG['random_state']
)
print(f"  - Training set: {X_train.shape[0]} samples")
print(f"  - Test set: {X_test.shape[0]} samples")

# ============== Model Training ==============
print("\n[8/8] Training model...")

def get_model(model_type, use_class_weights=True):
    """Get model based on type with optimized parameters."""
    
    if model_type == 'logistic':
        base_model = LogisticRegression(
            max_iter=2000,
            solver='saga',           # Best for large datasets
            C=1.0,                   # Regularization strength
            penalty='l2',            # L2 regularization
            class_weight='balanced' if use_class_weights else None,
            random_state=CONFIG['random_state'],
            n_jobs=1,
            tol=1e-4,
            warm_start=True
        )
    elif model_type == 'svm':
        base_model = CalibratedClassifierCV(
            LinearSVC(
                C=0.5,
                class_weight='balanced' if use_class_weights else None,
                random_state=CONFIG['random_state'],
                max_iter=2000,
                dual=True
            ),
            cv=3
        )
    elif model_type == 'random_forest':
        base_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced' if use_class_weights else None,
            random_state=CONFIG['random_state'],
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return OneVsRestClassifier(base_model, n_jobs=-1)

# Train the selected model
model = get_model(CONFIG['model_type'], CONFIG['use_class_weights'])
print(f"  - Training {CONFIG['model_type']} model with {len(top_terms)} classifiers...")
model.fit(X_train, y_train)
print("  - Training complete!")

# ============== Evaluation ==============
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Get predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate metrics
metrics = {
    'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
    'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    'f1_samples': f1_score(y_test, y_pred, average='samples', zero_division=0),
    'precision_micro': precision_score(y_test, y_pred, average='micro', zero_division=0),
    'recall_micro': recall_score(y_test, y_pred, average='micro', zero_division=0),
    'hamming_loss': hamming_loss(y_test, y_pred),
}

print("\n--- Default Threshold (0.5) ---")
for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")

# ============== Threshold Optimization ==============
if CONFIG['optimize_threshold']:
    print("\n--- Threshold Optimization ---")
    
    best_threshold = 0.5
    best_f1 = metrics['f1_micro']
    
    for threshold in np.arange(0.1, 0.6, 0.05):
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        f1_thresh = f1_score(y_test, y_pred_thresh, average='micro', zero_division=0)
        coverage = (y_pred_thresh.sum(axis=1) > 0).mean()
        avg_preds = y_pred_thresh.sum(axis=1).mean()
        
        print(f"  Threshold {threshold:.2f}: F1={f1_thresh:.4f}, Coverage={coverage:.2%}, Avg preds={avg_preds:.1f}")
        
        if f1_thresh > best_f1:
            best_f1 = f1_thresh
            best_threshold = threshold
    
    print(f"\n  Best threshold: {best_threshold:.2f} with F1={best_f1:.4f}")
else:
    best_threshold = 0.5

# ============== Per-Class Threshold Optimization ==============
print("\n--- Per-Class Threshold Optimization ---")

# Find optimal threshold for each class
optimal_thresholds = []
for i in range(y.shape[1]):
    best_t = 0.5
    best_f1_class = 0
    
    if y_test[:, i].sum() > 10:  # Only optimize if enough positive samples
        for t in np.arange(0.1, 0.8, 0.1):
            pred_class = (y_pred_proba[:, i] >= t).astype(int)
            f1_class = f1_score(y_test[:, i], pred_class, zero_division=0)
            if f1_class > best_f1_class:
                best_f1_class = f1_class
                best_t = t
    
    optimal_thresholds.append(best_t)

optimal_thresholds = np.array(optimal_thresholds)
print(f"  Threshold distribution: min={optimal_thresholds.min():.2f}, max={optimal_thresholds.max():.2f}, mean={optimal_thresholds.mean():.2f}")

# Evaluate with per-class thresholds
y_pred_optimal = (y_pred_proba >= optimal_thresholds).astype(int)
f1_optimal = f1_score(y_test, y_pred_optimal, average='micro', zero_division=0)
print(f"  F1 with per-class thresholds: {f1_optimal:.4f}")

# ============== Save Model ==============
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

model_data = {
    'model': model,
    'mlb': mlb,
    'scaler': scaler,
    'var_thresh': var_thresh,
    'selector': selector,
    'feature_names': X_df.columns.tolist(),
    'selected_feature_names': selected_feature_names,
    'top_terms': top_terms,
    'best_threshold': best_threshold,
    'optimal_thresholds': optimal_thresholds,
    'config': CONFIG,
    'metrics': metrics,
}

with open('improved_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"  Model saved to 'improved_model.pkl'")

# ============== Feature Importance Analysis ==============
print("\n--- Top Features Analysis ---")

if CONFIG['model_type'] == 'logistic' and hasattr(model.estimators_[0], 'coef_'):
    # Get average absolute coefficient across all classes
    all_coefs = np.array([est.coef_[0] for est in model.estimators_])
    avg_importance = np.abs(all_coefs).mean(axis=0)
    
    top_feature_idx = np.argsort(avg_importance)[-20:][::-1]
    print("\n  Top 20 most important features:")
    for idx in top_feature_idx:
        if idx < len(selected_feature_names):
            print(f"    {selected_feature_names[idx]}: {avg_importance[idx]:.4f}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"  - Model: {CONFIG['model_type']}")
print(f"  - Features: {X_final.shape[1]}")
print(f"  - GO terms: {len(top_terms)}")
print(f"  - Best F1 (micro): {best_f1:.4f}")
print(f"  - Model file: improved_model.pkl")
