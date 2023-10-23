# PredBERT

Implementation of the paper "Augmenting BERT-style models with predictive coding to improve discourse-level representations".

## Requirements and Installation

Install `requirements.txt` using Python 3.6.

Install `transformers` from the folder included in this repository.
```
git clone https://github.com/vgaraujov/PredBert.git
cd PredBert/transformers
pip install -e .
```

## Run Training

To run the continued pre-training of models, execute:
`python train_predalbert_r.py`

## Citation

```
@inproceedings{araujo-etal-2021-augmenting,
    title = "Augmenting {BERT}-style Models with Predictive Coding to Improve Discourse-level Representations",
    author = "Araujo, Vladimir  and
      Villa, Andr{\'e}s  and
      Mendoza, Marcelo  and
      Moens, Marie-Francine  and
      Soto, Alvaro",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.240",
    doi = "10.18653/v1/2021.emnlp-main.240",
    pages = "3016--3022",
}

@Article{make5010005,
AUTHOR = {Araujo, Vladimir and Moens, Marie-Francine and Soto, Alvaro},
TITLE = {Learning Sentence-Level Representations with Predictive Coding},
JOURNAL = {Machine Learning and Knowledge Extraction},
VOLUME = {5},
YEAR = {2023},
NUMBER = {1},
PAGES = {59--77},
URL = {https://www.mdpi.com/2504-4990/5/1/5},
ISSN = {2504-4990},
DOI = {10.3390/make5010005}
}
```
