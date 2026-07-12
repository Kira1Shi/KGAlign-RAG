# KGAlign-RAG

Repository for the project **"Using Knowledge Graph Alignment to Measure Answer Support in RAG"**.

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
````

### Test Graph Construction

```bash
python main.py
```

### Run Dataset Analysis

#### Download Datasets

```bash
mkdir -p data

cd data

git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git
git clone --depth 1 https://github.com/google-research-datasets/xsum_hallucination_annotations.git

cd ..
```

#### Run Analysis

```bash
python dataset_analysis/run_experiment.py extract_entities_method
```
