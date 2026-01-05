# HAN-Classifier

A Hierarchical Attention Network (HAN) for multi-class document classification on the 20 Newsgroups dataset.

## Overview

This project implements a custom deep learning architecture that combines contextual modeling, word-level filtering, sentence-level representation, and document-level attention for text classification. The model achieves **74.22% accuracy** on the 20 Newsgroups benchmark, outperforming traditional ML baselines.

## Architecture

![Architecture](assets/architecture.png)

The model processes documents through five interconnected stages:

1. **Word Embedding Layer** - Pre-trained GloVe embeddings (300d)
2. **Contextual Encoder** - 2-layer Bi-LSTM with residual connections and layer normalization
3. **Word-Level Attention** - Self-attention mechanism with top-k filtering (k=15)
4. **Sentence-Level Encoder** - Bi-LSTM with learned attention pooling
5. **Document-Level Cross-Attention** - 8-head multi-head attention with gating mechanism
6. **Classification Layer** - MLP classifier (512 → 256 → 20)

## Key Features

- **Dual-Stage Attention**: Word-level filtering + document-level cross-attention
- **Residual Connections**: Improved gradient flow through deep recurrent layers
- **Gating Mechanism**: Learnable fusion of sentence and cross-attention pathways
- **Top-K Word Filtering**: Noise reduction by retaining only important words
- **Comprehensive Analysis**: Attention visualization, error analysis, and failure mode identification

## Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 72.30% | 0.7305 | 0.7230 | 0.7250 |
| Naive Bayes | 72.76% | 0.7431 | 0.7276 | 0.7199 |
| Linear SVM | 72.08% | 0.7217 | 0.7208 | 0.7202 |
| **HAN (Ours)** | **74.22%** | **0.7495** | **0.7422** | **0.7446** |

### Key Findings

- Higher attention entropy correlates with correct predictions
- Cross-attention pathway contributes ~3% to final representation (gate weight ~0.031)
- Best performing class: `rec.sport.hockey` (92.5% accuracy)
- Most challenging class: `talk.religion.misc` (52.2% accuracy)

## Installation

```bash
# Clone the repository
git clone https://github.com/Anik81/HAN-Classifier.git
cd HAN-Classifier

# Install dependencies
pip install torch torchvision torchaudio
pip install scikit-learn nltk matplotlib seaborn tqdm pandas numpy gensim
```

## Usage

### Quick Start with Google Colab

1. Open `newsgroup_classification.ipynb` in Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Run all cells

### Local Training

```python
# Load and preprocess data
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

# Initialize model
model = HierarchicalAttentionNetwork(
    embedding_matrix=embedding_matrix,
    word_hidden_dim=256,
    sent_hidden_dim=256,
    attention_dim=128,
    num_classes=20,
    num_heads=8,
    dropout=0.4,
    top_k_words=15
)

# Train
python train.py --epochs 20 --batch_size 32 --lr 0.001
```

### Inference

```python
# Load trained model
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# Predict
text = "Your document text here..."
prediction = predict(model, text, preprocessor, device)
print(f"Predicted class: {TARGET_NAMES[prediction]}")
```

## Project Structure

```
HAN-Classifier/
├── newsgroup_classification.ipynb  # Main notebook (Colab-ready)
├── README.md
├── requirements.txt
├── assets/
│   ├── architecture.png
│   ├── confusion_matrix.png
│   ├── attention_statistics.png
│   └── failure_modes.png
├── outputs/
│   ├── best_model.pt
│   ├── model_comparison.csv
│   ├── per_class_metrics.csv
│   └── attention_statistics.csv
└── report/
    └── Newsgroup_Classification_Report.pdf
```

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 300 (GloVe) |
| Vocabulary Size | 30,467 |
| Word Encoder Hidden | 256 × 2 (bidirectional) |
| Sentence Encoder Hidden | 256 × 2 (bidirectional) |
| Attention Dimension | 128 |
| Cross-Attention Heads | 8 |
| Top-K Words | 15 |
| Dropout | 0.4 |
| Total Parameters | 17,538,712 |

## Dataset

The [20 Newsgroups dataset](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset) contains ~18,000 newsgroup posts across 20 categories:

- **Training samples**: 11,314
- **Test samples**: 7,532
- **Categories**: 20 (alt.atheism, comp.graphics, sci.space, etc.)

## Attention Analysis

The model provides interpretable attention weights at both word and sentence levels:

- **Word Attention**: Identifies important words within each sentence
- **Sentence Attention**: Weighs sentence importance for document representation
- **Cross-Attention**: Connects filtered words to relevant sentences

### Visualization Examples

```python
# Get attention weights for a sample
attn_data = get_attention_for_sample(model, document, label, preprocessor, device)
visualize_attention(attn_data)
```

## Failure Modes Identified

1. **Word Over-Concentration (8%)**: Attention fixates on single words
2. **Top-K Thresholding Errors (50%)**: Important words filtered out
3. **Local-Global Signal Conflict (16%)**: Short documents lack hierarchical context
4. **Diffuse Sentence Attention (50%)**: Attention spreads too evenly

## Future Improvements

- [ ] Replace Bi-LSTM with pre-trained Transformer (BERT/RoBERTa)
- [ ] Implement soft top-k filtering
- [ ] Add auxiliary cross-attention supervision loss
- [ ] Dynamic k selection based on attention entropy

## References

- [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf) (Yang et al., 2016)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)

## License

MIT License

## Author

**Tanvir Rahman Anik**  
📧 tranik.cse@gmail.com

---

⭐ Star this repository if you find it helpful!
