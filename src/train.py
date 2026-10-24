"""TF-IDF baseline and Hugging Face transformer fine-tuning."""
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .utils import SEED


class LabelSpace:
    def __init__(self, values):
        self.names = sorted(set(values))
        self.index = {n: i for i, n in enumerate(self.names)}

    def encode(self, values) -> np.ndarray:
        return np.array([self.index[v] for v in values])

    def __len__(self):
        return len(self.names)


# --- baseline ----------------------------------------------------------------

def tfidf_logreg(train_texts, y_train, eval_sets: dict, C: float = 4.0, ngram_max: int = 2,
                 max_features: int = 200_000, class_weight: Optional[str] = "balanced"):
    """Fit TF-IDF + multinomial logistic regression. Returns {split: decision scores}."""
    vec = TfidfVectorizer(ngram_range=(1, ngram_max), max_features=max_features, min_df=2,
                          sublinear_tf=True, strip_accents="unicode", lowercase=True)
    Xtr = vec.fit_transform(train_texts)
    clf = LogisticRegression(C=C, max_iter=300, class_weight=class_weight, random_state=SEED)
    t0 = time.time()
    clf.fit(Xtr, y_train)
    scores = {name: clf.decision_function(vec.transform(texts)) for name, texts in eval_sets.items()}
    return scores, time.time() - t0


# --- transformers --------------------------------------------------------------

@dataclass
class TrainConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 256
    learning_rate: float = 2e-5
    batch_size: int = 32  # effective batch size across all GPUs
    epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    loss: str = "weighted_ce"  # "ce" | "weighted_ce" | "focal"
    focal_gamma: float = 2.0
    dropout: Optional[float] = None  # None keeps the pretrained model's default (0.1 for BERT)
    early_stopping_patience: int = 1
    seed: int = SEED


def _make_trainer_cls():
    from transformers import Trainer

    class LossTrainer(Trainer):
        """Trainer with class-weighted cross-entropy or focal loss."""

        def __init__(self, *args, loss="weighted_ce", class_weights=None, focal_gamma=2.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_name = loss
            self.class_weights = class_weights
            self.focal_gamma = focal_gamma

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits.float()
            w = self.class_weights.to(logits.device) if self.loss_name != "ce" else None
            if self.loss_name == "focal":
                logp = torch.log_softmax(logits, -1)
                logp_t = logp.gather(1, labels[:, None]).squeeze(1)
                loss = -((1 - logp_t.exp()) ** self.focal_gamma) * logp_t
                if w is not None:
                    loss = loss * w[labels]
                loss = loss.mean()
            else:
                loss = torch.nn.functional.cross_entropy(logits, labels, weight=w)
            return (loss, outputs) if return_outputs else loss

    return LossTrainer


def train_transformer(cfg: TrainConfig, train_texts, y_train, eval_sets: dict, n_classes: int,
                      output_dir: str, callbacks=(), select_on: str = "val"):
    """Fine-tune `cfg.model_name`. `eval_sets` maps split name -> (texts, labels); the `select_on`
    split drives early stopping / best-checkpoint selection. Returns (scores per split, info)."""
    from datasets import Dataset
    from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, EarlyStoppingCallback, TrainingArguments, set_seed)

    set_seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    config = AutoConfig.from_pretrained(cfg.model_name, num_labels=n_classes)
    if cfg.dropout is not None:
        for attr in ("hidden_dropout_prob", "attention_probs_dropout_prob", "dropout", "attention_dropout",
                     "seq_classif_dropout", "classifier_dropout"):
            if hasattr(config, attr):
                setattr(config, attr, cfg.dropout)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=config)

    def to_ds(texts, labels):
        ds = Dataset.from_dict({"text": list(texts), "label": np.asarray(labels, dtype=np.int64)})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=cfg.max_length),
                      batched=True, remove_columns=["text"])

    train_ds = to_ds(train_texts, y_train)
    eval_ds = {name: to_ds(t, y) for name, (t, y) in eval_sets.items()}

    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / (n_classes * np.clip(counts, 1, None)))

    n_gpu = max(torch.cuda.device_count(), 1)
    per_device = max(cfg.batch_size // n_gpu, 1)
    steps = int(np.ceil(len(train_ds) / (per_device * n_gpu))) * cfg.epochs

    def compute_metrics(p):
        from sklearn.metrics import accuracy_score, f1_score
        pred = p.predictions.argmax(-1)
        return {"macro_f1": f1_score(p.label_ids, pred, average="macro"),
                "accuracy": accuracy_score(p.label_ids, pred)}

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=per_device * 4,
        num_train_epochs=cfg.epochs,
        warmup_steps=int(cfg.warmup_ratio * steps),
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        seed=cfg.seed,
        dataloader_num_workers=2,
        disable_tqdm=True,
    )
    trainer = _make_trainer_cls()(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds[select_on],
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(cfg.early_stopping_patience), *callbacks],
        loss=cfg.loss,
        class_weights=class_weights,
        focal_gamma=cfg.focal_gamma,
    )
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    scores = {name: trainer.predict(ds).predictions for name, ds in eval_ds.items()}
    history = [h for h in trainer.state.log_history if "eval_macro_f1" in h]
    info = {"config": asdict(cfg), "train_seconds": train_time, "eval_history": history,
            "best_epoch": max(history, key=lambda h: h["eval_macro_f1"])["epoch"] if history else None}
    return scores, info

