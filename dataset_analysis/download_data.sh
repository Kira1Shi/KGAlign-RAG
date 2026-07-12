#!/usr/bin/env bash
set -euo pipefail

mkdir -p data && cd data

curl -sL -o ragtruth.zip https://codeload.github.com/ParticleMedia/RAGTruth/zip/refs/heads/main
unzip -oq ragtruth.zip && rm ragtruth.zip
rm -rf RAGTruth && mv RAGTruth-main RAGTruth

curl -sL -o xsum.zip https://codeload.github.com/google-research-datasets/xsum_hallucination_annotations/zip/refs/heads/master
unzip -oq xsum.zip && rm xsum.zip
rm -rf xsum_hallucination_annotations && mv xsum_hallucination_annotations-master xsum_hallucination_annotations

python3 -c "import nltk; [nltk.download(p, quiet=True) for p in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger_eng', 'maxent_ne_chunker', 'maxent_ne_chunker_tab', 'words']]"
