# NLP Portfolio

Eight natural language processing projects covering the pipeline from text preprocessing to neural machine translation. Each project is a self-contained Jupyter notebook with a baseline, held-out evaluation, error analysis, and a short write-up of findings and limitations.


## Projects

| # | Project | Techniques | Headline result |
|---|---|---|---|
| 01 | [Linguistic Profiling of a Short Story with spaCy](01_Text_Preprocessing/spacy_linguistic_profile_owl_creek_bridge.ipynb) | Tokenization, lemmatization vs. stemming, NER, `Matcher` / `PhraseMatcher`, morphology | Verb-tense tags detect Bierce's shift to present tense: 3.8% → 51.9% of finite verbs in the final scene |
| 02a | [Yelp Polarity: Preprocessing and Representation Study](02_Text_Classification/yelp_polarity_preprocessing_and_representation_study.ipynb) | BoW vs. TF-IDF, stop-word/stemming ablation, n-grams, logistic regression | 94.7% test accuracy; standard stop-word removal *lowers* accuracy by 1.3 points |
| 02b | [Movie Review Polarity: TF-IDF From Scratch](02_Text_Classification/movie_review_polarity_tfidf_from_scratch.ipynb) | NumPy/SciPy TF-IDF verified against scikit-learn, cross-validated model comparison, learning curve | 88.5% test accuracy (tuned linear SVM); from-scratch TF-IDF matches scikit-learn to 4e-16 |
| 03 | [Lexicon vs. Supervised Sentiment Analysis](03_Sentiment_Analysis/movie_review_sentiment_vader_vs_tfidf.ipynb) | VADER, threshold tuning, TF-IDF + logistic regression, sentence-level analysis | TF-IDF model 93.1% vs. VADER 77.1%; disproves the "verdict is in the last sentence" explanation |
| 04 | [POS and NER Benchmark: NLTK vs. spaCy](04_Information_Extraction/pos_ner_benchmark_nltk_vs_spacy.ipynb) | CoNLL-2003 evaluation, entity-level F1 from scratch, error taxonomy, NER + VADER pipeline | spaCy NER F1 0.595 vs. NLTK 0.477; tag-convention harmonization reverses the raw POS ranking |
| 05a | [Word-Level LSTM Language Model on *Moby-Dick*](05_Text_Generation/word_level_lstm_language_model_moby_dick.ipynb) | PyTorch LSTM, truncated BPTT, weight tying, n-gram baselines, decoding strategies | Test perplexity 115.6 vs. 169.0 for an interpolated trigram |
| 05b | [Character-Level RNN vs. LSTM vs. GRU](05_Text_Generation/char_level_rnn_lstm_gru_comparison.ipynb) | PyTorch, bits per character, n-gram baselines, context-length analysis | GRU and LSTM tie at 1.90 test bits/char vs. 2.06 for a vanilla RNN; GRU uses 25% fewer parameters |
| 06 | [English→Japanese NMT with Attention](06_Machine_Translation/english_japanese_seq2seq_attention.ipynb) | PyTorch encoder-decoder, Luong attention, retrieval baseline, BLEU/chrF | Attention improves chrF from 0.214 to 0.261 and beats the retrieval baseline (0.240) |

## Running the notebooks

Each project folder has its own `requirements.txt`. A single environment with Python 3.10+ covers all of them:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pandas numpy scipy scikit-learn nltk spacy "datasets>=2.14,<4" matplotlib seaborn jupyter
pip install torch                  # for a CUDA build, see https://pytorch.org/get-started/locally/
python -m spacy download en_core_web_sm
jupyter lab
```

Run each notebook from inside its own folder, because data paths are relative (`data/...`). NLTK resources and Hugging Face datasets download automatically on first use.

| Project | Data | Approximate runtime |
|---|---|---|
| 01 | Included (`data/owlcreek.txt`, public domain) | < 1 min |
| 02a | Downloads Yelp Polarity from Hugging Face (about 160 MB) | about 5 min |
| 02b | Downloads NLTK `movie_reviews` | about 1 min |
| 03 | **Not included.** Place the course file at `data/movie_reviews.tsv` (see the notebook) | < 1 min |
| 04 | Downloads CoNLL-2003 and IMDb from Hugging Face | about 3 min |
| 05a | Included (`data/melville-moby_dick.txt`, public domain) | about 2 min on GPU |
| 05b | Streams a 5,000-document sample of OpenWebText | about 15 min on GPU (reduced budget on CPU) |
| 06 | Included (`data/english_japanese.csv`, CC BY 4.0) | about 3 min on GPU |

Runtimes were measured on an NVIDIA RTX 3060 Ti and a 16 Gb RAM. The PyTorch notebooks fall back to the CPU automatically, but they are much slower there.

## Repository layout

```
NLP_Portfolio/
├── 01_Text_Preprocessing/
├── 02_Text_Classification/
├── 03_Sentiment_Analysis/
├── 04_Information_Extraction/
├── 05_Text_Generation/
└── 06_Machine_Translation/
```

Each folder contains the notebook(s), a `requirements.txt`, and a `data/` folder when the data can be redistributed.
