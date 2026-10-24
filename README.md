# arXiv research-article classification: how many categories, and how well can we predict them?

Classify arXiv papers from their abstracts into an **unknown** number of subject categories, find the
best number of categories, and train a Hugging Face transformer to predict them.

<!-- RESULTS:TLDR -->

## Contents

1. [Overview and objective](#1-overview-and-objective)
2. [Literature review](#2-literature-review)
3. [Dataset and EDA](#3-dataset-and-eda)
4. [Preprocessing](#4-preprocessing)
5. [Discovering the number of categories](#5-discovering-the-number-of-categories)
6. [Models and training](#6-models-and-training)
7. [Evaluation](#7-evaluation)
8. [Model selection and hyperparameter tuning](#8-model-selection-and-hyperparameter-tuning)
9. [Conclusions](#9-conclusions)
10. [Limitations and future work](#10-limitations-and-future-work)
11. [How to reproduce](#11-how-to-reproduce)

---

## 1. Overview and objective

arXiv labels every paper with one primary category from a three-level taxonomy:
**8 groups** (e.g. Physics) → **20 archives** (e.g. `hep-th`, `cs`) → **~150 fine categories**
(e.g. `cs.LG`). The taxonomy was designed by people for a submission system, so it is not obvious that
it is the best set of classes for *predicting a subject from the text*: some categories are
near-duplicates in content, others are broad.

So this project answers two questions:

1. **How many categories do abstracts really support?** We find out from the text, without using the
   labels, and then check the answer against arXiv's own labels.
2. **How well can a fine-tuned BERT model predict them**, compared with a classic TF-IDF baseline,
   and how much does tuning help?

## 2. Literature review

The three papers this implementation is built on:

| # | Paper | What we take from it |
|---|---|---|
| 1 | **Devlin, Chang, Lee & Toutanova (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT.** [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) | The classifier: a pretrained bidirectional Transformer encoder with a softmax layer on the `[CLS]` token, fine-tuned end-to-end. We use the `bert-base-uncased` checkpoint (110M parameters) from Hugging Face. |
| 2 | **Sun, Qiu, Xu & Huang (2019). *How to Fine-Tune BERT for Text Classification?* CCL.** [arXiv:1905.05583](https://arxiv.org/abs/1905.05583) | The fine-tuning recipe: small learning rate (2e-5), warmup with linear decay, 2–4 epochs, and truncation to 256–512 tokens. It is our default configuration and the centre of our search space. |
| 3 | **Cohan, Feldman, Beltagy, Downey & Weld (2020). *SPECTER: Document-level Representation Learning using Citation-informed Transformers.* ACL.** [arXiv:2004.07180](https://arxiv.org/abs/2004.07180) (SPECTER2: Singh et al., 2023, *SciRepEval*, EMNLP, [arXiv:2211.13308](https://arxiv.org/abs/2211.13308)) | Paper embeddings learned from citations, not from arXiv categories. We cluster SPECTER2 embeddings of the abstracts to find the number of categories, so this step does not depend on the labels we later predict. |

**Supporting references.**

- Beltagy, Lo & Cohan (2019), *SciBERT*, [arXiv:1903.10676](https://arxiv.org/abs/1903.10676). This paper shows that scientific text differs enough from general text that domain pretraining helps, which is why we embed with a science-specific encoder.
- Sanh et al. (2019), *DistilBERT*, [arXiv:1910.01108](https://arxiv.org/abs/1910.01108). A 40% smaller BERT, one of our model-selection candidates.
- Clement et al. (2019), *On the Use of ArXiv as a Dataset*, [arXiv:1905.00075](https://arxiv.org/abs/1905.00075). The pipeline behind the Kaggle snapshot, including the category taxonomy.
- Ward (1963), *Hierarchical grouping to optimize an objective function*, JASA. The linkage used to merge categories.
- Rousseeuw (1987), *Silhouettes*, J. Comput. Appl. Math. Davies & Bouldin (1979), IEEE TPAMI. The cluster-validity indices.
- Satopää et al. (2011), *Finding a "Kneedle" in a Haystack*, ICDCS-W. Knee detection on the metric-vs-k curve.
- Lin et al. (2017), *Focal Loss for Dense Object Detection*, ICCV. Focal loss, one of the tuning choices.
- Akiba et al. (2019), *Optuna*, KDD. TPE search with median pruning.

**Why this setup.** Fine-tuned encoders have been the standard strong baseline for topic
classification since BERT. The task asks for a Hugging Face BERT-family model with TensorFlow weights
and an Apache-2.0 licence; `bert-base-uncased`, `bert-base-cased` and `distilbert-base-uncased` all
qualify. The models are trained with PyTorch, because current `transformers` releases (v5) no longer
include the TensorFlow back-end; the checkpoints are the same.

<!-- RESULTS:EDA -->

<!-- RESULTS:PREPROCESSING -->

<!-- RESULTS:DISCOVERY -->

<!-- RESULTS:TRAINING -->

<!-- RESULTS:TUNING -->

<!-- RESULTS:CONCLUSIONS -->

## 11. How to reproduce

All heavy computation runs on **Kaggle** (free CPU/GPU); the local machine only edits code.

```
├── notebooks/        01_eda_preprocessing · 02_discover_k · 03_train_evaluate · 04_tuning
│                     (.py = source in jupytext "percent" format, .ipynb = executed copy with outputs)
├── src/              preprocess · taxonomy · discover_k · train · evaluate · tune · viz · utils
├── reports/figures/  every figure in this README
├── results/          metrics tables (CSV/JSON) behind every number in this README
└── kaggle/run.py     runs a notebook on Kaggle and downloads its outputs
```

**On Kaggle, by hand.** Create a notebook, attach the dataset `Cornell-University/arxiv`, upload
`src/` as a dataset, paste in a notebook from `notebooks/`, and run. Notebooks 02–04 need
*Accelerator: GPU T4 ×2* and *Internet: on* (to download models from the Hugging Face Hub). Each
notebook reads the previous notebooks' outputs, so attach those as inputs.

**From this repo, automated.** Requires a Kaggle account with a verified phone number (needed for
GPU and internet) and an API token in `~/.kaggle/access_token`:

```bash
pip install kaggle jupytext
python kaggle/run.py push 01          && python kaggle/run.py pull 01   # CPU, ~15 min
python kaggle/run.py push 02 --gpu    && python kaggle/run.py pull 02   # ~30 min
python kaggle/run.py push 03 --gpu    && python kaggle/run.py pull 03   # ~2 h
python kaggle/run.py push 04 --gpu    && python kaggle/run.py pull 04   # ~4 h
```

`push` uploads `src/` and the notebooks as a private Kaggle dataset and starts a private Kaggle
notebook. `pull` waits for it to finish, then copies the figures, results tables and the executed
notebook back into the repo. Seeds are fixed (42) throughout. Library versions are pinned in
`requirements.txt` to match the Kaggle image the results were produced with.
