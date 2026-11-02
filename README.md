# Clickbait & Misinformation Detection with Interpretability

A machine learning project focused on detecting clickbait and false claims in news articles and social media posts, with a strong emphasis on explaining predictions. The goal is to analyze the semantic gap between headlines and target article content while ensuring the models remain fully interpretable and robust.

## Dataset

The project uses the [Webis Clickbait Corpus 2017 (Webis-Clickbait-17)](https://zenodo.org/records/5530410) dataset, which contains 38,517 Twitter posts from major US news publishers. The data covers:

- **Post info** - Twitter post text (headline) and attached media
- **Article info** - target article title, description, and paragraphs
- **Target** - 4-point scale truth judgments (from 0.0 to 1.0) assessed by 5 independent annotators

*Note: The dataset is excluded from version control. See the Setup section for download instructions.*

## Models

We train and compare different approaches ranging from a classical baseline to deep neural networks:

TBD

### Key Metric & Interpretability

> **Macro F1-Score & Model Faithfulness** - [TBD].

All experiments are tracked with **MLflow** for reproducibility and performance comparison[cite: 6].

## Project Structure

```text
├── data/
│   ├── raw/                      # Extracted JSONL and images (downloaded manually)
│   ├── interim/                  # Intermediate text transformations
│   └── processed/                # Final features and embeddings
├── notebooks/
│   ├── 01_data_exploration.ipynb # Exploratory data analysis (EDA)
│   ├── 02_features_and_baseline.ipynb # Classical features and baseline model[cite: 8]
│   ├── 03_deep_models.ipynb      # Neural networks and transformers[cite: 8]
│   ├── 04_evaluation_and_interpretability.ipynb # XAI (SHAP/LIME) and error analysis[cite: 8]
│   └── 05_final_report.ipynb     # End-to-end report with all results[cite: 8]
├── src/
│   ├── __init__.py
│   ├── data.py                   # Data loading, parsing JSONL, and splits[cite: 8]
│   ├── features.py               # Feature engineering (stylometric, TF-IDF)[cite: 8]
│   ├── models.py                 # Model definitions and architectures[cite: 8]
│   └── train.py                  # Training loops and evaluation functions[cite: 8]
├── configs/                      # Hyperparameter configurations[cite: 8]
├── mlruns/                       # MLflow tracking database and artifacts[cite: 8]
├── reports/                      
│   └── figures/                  # Generated plots and XAI visualizations[cite: 8]
├── requirements.txt              # Project dependencies[cite: 8]
└── README.md
```

## Setup

```bash
# Create and activate a virtual environment
python -m venv clickbait_project
clickbait_project\Scripts\activate      # Windows (Git Bash)
# source clickbait_project/bin/activate # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

## Usage

TBD