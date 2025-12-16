# CAFA-6 Protein Function Prediction

## Project Overview

This repository contains multiple machine learning models for the **CAFA-6 (Critical Assessment of Functional Annotation) Protein Function Prediction** competition on Kaggle. The goal is to predict Gene Ontology (GO) terms for proteins based on their amino acid sequences.

### Competition Details
- **Dataset**: 82,404 training proteins, 224,309 test proteins
- **Task**: Multi-label classification (predict multiple GO terms per protein)
- **Target**: 26,125 unique GO terms describing protein functions
- **Evaluation**: F-max score with Information Accretion (IA) weights

## Models Overview

This repository implements **four different approaches** to protein function prediction, plus an **ensemble/post-processing tool**:

| Model | Features | Performance | Speed | Recommended For |
|-------|----------|-------------|-------|------------------|
| **MLP + Resnet** | ESM-2 embeddings (1280D) | Best | Best | Primary model - highest accuracy |
| **KNN** | ESM-2 embeddings (1280D) | Very Good | Good | Deep learning approach |
| **Logistic Regression** | Hand-crafted features (62D) | Baseline | Medium | Quick baseline, feature engineering |

## Repository Structure

```
cafa-6-protein-function-prediction/
├── src/
│   ├── KNN/
│   │   └── ESM2-KNN.ipynb                    # PRIMARY MODEL
│   ├── MLP + Resnet/
│   │   └── ESM2-Resnet.ipynb                 # Deep learning model
│   ├── Logistic Regression/
│   │   └── logistic_regression_model.ipynb   # Baseline model
│   ├── Embedding Generator/
│   │   └── Embedding.ipynb                   # ESM-2 embedding generation
│   └── merge-submit.ipynb                    # ENSEMBLE & POST-PROCESSING
├── Train/                                     # Training data
│   ├── train_sequences.fasta
│   ├── train_terms.tsv
│   ├── train_taxonomy.tsv
│   └── go-basic.obo
├── Test/                                      # Test data
│   ├── testsuperset.fasta
│   └── testsuperset-taxon-list.tsv
├── IA.tsv                                     # Information Accretion weights
├── sample_submission.tsv                      # Submission format example
└── README.md
```

---

## 📂 Models Description

### 1. KNN (K-Nearest Neighbors) - **RECOMMENDED**
**Location**: `src/KNN/ESM2-KNN.ipynb`

#### What it does:
- Uses **pre-computed ESM-2 embeddings** (1280-dimensional protein representations)
- Finds K most similar proteins using **GPU-accelerated cosine similarity**
- Propagates GO term labels from similar proteins with weighted voting
- Searches for optimal K value (10-100) using validation set

#### Key Features:
- **GPU Acceleration**: PyTorch-based cosine similarity (10-20x faster than CPU)
- **Optimal K Search**: Validates K=[10,20,30,50,70,100] to find best performance
- **Weighted Label Propagation**: `Score(Term) = Σ(Similarity × Label) / Σ(Similarity)`
- **Memory Efficient**: Batch processing for 16GB VRAM constraint
- **Best Performance**: Expected 2-3x F1 improvement over baseline

#### How to Use:
1. Ensure ESM-2 embeddings are available on Kaggle:
   - `train_embeddings.pt` (1280D embeddings for training proteins)
   - `test_embeddings.pt` (1280D embeddings for test proteins)
2. Open notebook in Kaggle with **GPU enabled** (P100 or T4)
3. Run all cells sequentially
4. Output: `submission.tsv` with predictions

#### Requirements:
- PyTorch
- NumPy, Pandas
- Pre-computed ESM-2 embeddings
- GPU recommended (CPU will be very slow)

---

### 2. MLP + ResNet (Deep Neural Network)
**Location**: `src/MLP + Resnet/ESM2-Resnet.ipynb`

#### What it does:
- Trains a **deep neural network** on ESM-2 embeddings
- Uses **ResNet-style architecture** with skip connections
- Multi-label binary classification (one output per GO term)
- Optimizes for F-max metric with custom loss function

#### Key Features:
- **Architecture**: Input (1280D) → Dense layers → ResNet blocks → Output (26k terms)
- **Skip Connections**: Prevents vanishing gradients, enables deeper networks
- **Batch Normalization**: Stabilizes training, faster convergence
- **Dropout**: Prevents overfitting on training data
- **Custom Loss**: Focal loss or weighted BCE for class imbalance

#### How to Use:
1. Ensure ESM-2 embeddings are available
2. Open notebook in Kaggle with GPU enabled
3. Adjust hyperparameters (learning rate, batch size, epochs) if needed
4. Run training cells (may take 1-2 hours)
5. Output: `submission.tsv` and trained model weights

#### Requirements:
- PyTorch or TensorFlow
- Pre-computed ESM-2 embeddings
- GPU strongly recommended

---

### 3. Logistic Regression (Baseline Model)
**Location**: `src/Logistic Regression/logistic_regression_model.ipynb`

#### What it does:
- Extracts **hand-crafted features** from amino acid sequences
- Trains **multi-label logistic regression** (600 most common GO terms)
- Fast baseline model using traditional machine learning

#### Key Features:
- **Feature Engineering** (62 features):
  - Amino acid composition (20 features)
  - Physicochemical properties (10 features: hydrophobic, polar, charged, etc.)
  - Dipeptide frequencies (20 features)
  - N-terminal and C-terminal composition (16 features)
  - Sequence length features (2 features)
- **GPU Acceleration**: Optional CuML/RAPIDS support for 5-10x speedup
- **Feature Standardization**: StandardScaler for better convergence
- **Class Balancing**: Handles imbalanced GO term distribution

#### How to Use:
1. Open notebook in Kaggle (GPU optional but recommended)
2. No embedding files needed - works directly with FASTA sequences
3. Run all cells sequentially
4. Training time: ~5-10 minutes on GPU, ~30-60 minutes on CPU
5. Output: `submission.tsv` and `logistic_regression_model.pkl`

#### Requirements:
- scikit-learn (CPU) or CuML (GPU)
- NumPy, Pandas
- No pre-computed embeddings needed

#### When to Use:
- Quick baseline to establish performance floor
- Feature engineering exploration
- No access to ESM-2 embeddings
- Limited computational resources

---

### 4. Embedding Generator (Preprocessing)
**Location**: `src/Embedding Generator/Embedding.ipynb`

#### What it does:
- Generates **ESM-2 embeddings** from raw protein FASTA sequences
- Uses Facebook's **ESM-2** pre-trained transformer model
- Outputs 1280-dimensional embedding vectors per protein

#### Key Features:
- **Pre-trained Model**: ESM-2 (650M parameters, trained on 250M proteins)
- **Contextualized Embeddings**: Captures protein structure and function patterns
- **Batch Processing**: Handles large datasets efficiently
- **GPU Required**: Very slow on CPU (hours vs minutes)

#### How to Use:
1. Open notebook in Kaggle with **GPU enabled** (required)
2. Point to FASTA files:
   - `Train/train_sequences.fasta`
   - `Test/testsuperset.fasta`
3. Run embedding generation (may take 1-3 hours for full dataset)
4. Output:
   - `train_embeddings.pt` (~1GB)
   - `test_embeddings.pt` (~3GB)

#### Requirements:
- Transformers library (`huggingface/transformers`)
- PyTorch
- ESM-2 model (downloaded automatically)
- GPU with 16GB+ VRAM
#### Important Notes:
- **Only needed once** - embeddings can be reused
- For Kaggle competition, embeddings are pre-computed and available as dataset
- Very computationally expensive - use existing embeddings when possible

---

### 5. Merge & Submit (Ensemble + Post-Processing) - **FINAL STEP**
**Location**: `src/merge-submit.ipynb`

#### What it does:
- **Ensembles multiple model predictions** with weighted averaging
#### Option 1: KNN Model + Ensemble (Recommended for Best Score)
```bash
1. Upload to Kaggle notebook
2. Add datasets:
   - cafa-5-protein-function-prediction (competition data)
   - cafa5-esm2-embeddings (pre-computed embeddings)
   - protein-go-annotations OR newgoa (UniProt GOA database)
3. Enable GPU accelerator
4. Open: src/KNN/ESM2-KNN.ipynb
5. Run all cells → saves submission_knn.tsv
6. Open: src/MLP + Resnet/ESM2-Resnet.ipynb (optional)
7. Run all cells → saves submission_resnet.tsv
8. Open: src/merge-submit.ipynb
9. Configure paths to point to your submission files
10. Run all cells → final submission.tsv
11. Submit: submission.tsv
```*Memory Efficient**: Processes large files with garbage collection

#### How to Use:
1. **Configure paths** in the Config class:
   ```python
   BEST_SUB = '/kaggle/input/.../submission_best.tsv'  # Your best model
   SECOND_SUB = '/kaggle/input/.../submission_knn.tsv' # Optional second model
   W_BEST = 0.9      # Weight for best model
   W_SECOND = 0.1    # Weight for second model
   GOA_PATH = '/kaggle/input/newgoa/goa_uniprot_all.csv'  # UniProt GOA database
   ```

2. **Add required datasets** on Kaggle:
   - Your model prediction files (upload as datasets)
   - UniProt GOA database (search "protein-go-annotations" or "newgoa")

3. Run all cells sequentially

4. Output: `submission.tsv` (ensemble + filtered + boosted)

#### Processing Steps:
1. **Parse GO Ontology**: Extract parent-child relationships from `go-basic.obo`
2. **Load GOA Database**: Extract positive and negative annotations
3. **Ensemble Models**: Weighted average of multiple predictions
4. **Remove Negatives**: Filter out protein-term pairs known to be incorrect
5. **Inject Positives**: Add/boost ground truth annotations to score 1.0
6. **Sanity Check**: Visualize confidence score distribution

#### Requirements:
- Pandas, NumPy
- Access to UniProt GOA database
- Pre-generated submission files from other models

#### When to Use:
- **Always use as final step** before competition submission
- Combining multiple models for better results
- Leveraging external knowledge (UniProt GOA)
- Maximizing competition score with known annotations

#### Expected Improvements:
- +5-15% F-max score improvement
- Better recall (adds missing true positives)
- Better precision (removes false positives)
- More consistent predictions (respects GO hierarchy)

#### Important Notes:
- **Requires UniProt GOA database** - must be added as Kaggle dataset
- Careful with data leakage - only use for competition with explicit permission
- Processing time: 5-15 minutes depending on file sizes
- Includes visualization for sanity checking output

---
---

## 🚀 Quick Start Guide

### For Kaggle Competition:

#### Option 1: KNN Model (Recommended)
```bash
1. Upload to Kaggle notebook
2. Add datasets:
   - cafa-5-protein-function-prediction (competition data)
   - cafa5-esm2-embeddings (pre-computed embeddings)
3. Enable GPU accelerator
4. Open: src/KNN/ESM2-KNN.ipynb
5. Run all cells
6. Submit: submission.tsv
```

#### Option 2: Logistic Regression (Fast Baseline)
```bash
1. Upload to Kaggle notebook
2. Add dataset: cafa-5-protein-function-prediction
3. GPU optional (but recommended)
4. Open: src/Logistic Regression/logistic_regression_model.ipynb
5. Run all cells
6. Submit: submission.tsv
```

### For Local Development:

```bash
# Clone repository
git clone <repository-url>
cd cafa-6-protein-function-prediction

# Install dependencies
pip install torch numpy pandas scikit-learn tqdm transformers

# Option 1: Generate embeddings (if not available)
jupyter notebook "src/Embedding Generator/Embedding.ipynb"

# Option 2: Use pre-computed embeddings from Kaggle
# Download from: https://www.kaggle.com/datasets/.../cafa5-esm2-embeddings

# Run KNN model
jupyter notebook "src/KNN/ESM2-KNN.ipynb"
## 🎯 Tips for Best Results

1. **Start with KNN**: Highest performance with minimal effort
2. **Use GPU**: All models benefit significantly from GPU acceleration
3. **Validate Properly**: Use stratified split, respect temporal order if available
4. **Tune Thresholds**: Each model has optimal prediction threshold
5. **Ensemble Models**: Use `merge-submit.ipynb` to combine predictions from multiple models
6. **Apply Post-Processing**: Always use ensemble + GOA filtering as final step
7. **Monitor Memory**: Batch processing prevents OOM errors
8. **Check Data Leakage**: Ensure no test proteins in training set
9. **Leverage External Data**: UniProt GOA database provides valuable filtering
10. **Visualize Results**: Use built-in sanity checks to verify submission qualityime |
|-------|-------|-----------|--------|---------------|----------------|
| KNN (K=50) | ~0.45-0.55 | ~0.50-0.60 | ~0.40-0.50 | None (KNN) | 10-15 min |
| MLP + ResNet | ~0.40-0.50 | ~0.45-0.55 | ~0.35-0.45 | 1-2 hours | 5-10 min |
| Logistic Regression | ~0.15-0.25 | ~0.20-0.30 | ~0.12-0.18 | 5-10 min | 2-5 min |

*Note: Actual scores vary based on hyperparameter tuning and validation strategy*

---

## 🔧 Technical Requirements

### Minimum Requirements:
- Python 3.8+
- 16GB RAM
- 50GB disk space

### Recommended for Full Pipeline:
- Python 3.10+
- 32GB RAM
- GPU with 16GB VRAM (NVIDIA P100/T4/V100)
- 100GB disk space
- CUDA 11.0+

### Python Packages:
```
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
cuml>=23.0.0 (optional, for GPU logistic regression)
cupy-cuda11x>=12.0.0 (optional, for GPU logistic regression)
```

---

## 📈 Model Selection Guide

**Choose KNN if:**
- ✅ You want the best performance
- ✅ You have access to pre-computed ESM-2 embeddings
- ✅ You have GPU available
- ✅ You're okay with no training time (model-free approach)

**Choose MLP + ResNet if:**
- ✅ You want deep learning approach
- ✅ You want to fine-tune on your specific data
- ✅ You have time for training (1-2 hours)
**Use Embedding Generator if:**
- ✅ You need to generate new embeddings
- ✅ You have custom protein sequences
- ✅ Pre-computed embeddings are not available

**Use Merge & Submit if:**
- ✅ You want to maximize competition score
- ✅ You have multiple model predictions to combine
- ✅ You want to leverage UniProt GOA database
- ✅ You need GO hierarchy consistency
- ✅ You're making your final submission

--- You don't have access to embeddings
- ✅ You have limited computational resources
- ✅ You want explainable predictions

**Use Embedding Generator if:**
- ✅ You need to generate new embeddings
- ✅ You have custom protein sequences
- ✅ Pre-computed embeddings are not available

---

## 🎯 Tips for Best Results

1. **Start with KNN**: Highest performance with minimal effort
2. **Use GPU**: All models benefit significantly from GPU acceleration
3. **Validate Properly**: Use stratified split, respect temporal order if available
4. **Tune Thresholds**: Each model has optimal prediction threshold
5. **Ensemble Models**: Combine predictions from multiple models for better results
6. **Monitor Memory**: Batch processing prevents OOM errors
7. **Check Data Leakage**: Ensure no test proteins in training set

---

## 📝 Citation

If you use this code in your research, please cite:

```
CAFA-6 Protein Function Prediction
Kaggle Competition: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview
ESM-2 Model: https://github.com/facebookresearch/esm
```

---

## 📧 Contact & Support

For questions or issues:
- Open an issue on GitHub
- Check Kaggle discussion forum
- Review model notebooks for detailed documentation

---

## 🔄 Version History

- **v1.0**: Initial KNN, MLP+ResNet, and Logistic Regression models
- **v1.1**: Added GPU acceleration to Logistic Regression
- **v1.2**: Optimized batch processing for memory efficiency
- **v2.0**: Complete refactor with comprehensive documentation

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
