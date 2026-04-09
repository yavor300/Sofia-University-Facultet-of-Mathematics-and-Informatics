from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from .data import build_option_level_frame

WORD_RE = re.compile(r"[A-Za-z]{2,}")


class LexicalOverlapRanker:
    """Unsupervised baseline that picks the option with max lexical overlap."""

    @staticmethod
    def _tokenize(text: str) -> set:
        return {token.lower() for token in WORD_RE.findall(text)}

    def predict(self, questions_df: pd.DataFrame) -> Dict[str, List[str]]:
        predictions: Dict[str, List[str]] = {}
        for _, row in questions_df.iterrows():
            q_tokens = self._tokenize(row["question"])
            best_label = row["choice_labels"][0]
            best_score = -1.0
            for label in row["choice_labels"]:
                option_text = row["option_texts"][label]
                o_tokens = self._tokenize(option_text)
                if not o_tokens:
                    score = 0.0
                else:
                    score = len(q_tokens.intersection(o_tokens)) / len(o_tokens)
                if score > best_score:
                    best_score = score
                    best_label = label
            predictions[row["id"]] = [best_label]
        return predictions


class OptionPairClassifier:
    """Supervised baseline: binary classification on (question, option) pairs."""

    def __init__(
        self,
        max_features: int = 40000,
        ngram_min: int = 1,
        ngram_max: int = 2,
        c_value: float = 1.0,
    ) -> None:
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=(ngram_min, ngram_max),
                        lowercase=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=800,
                        C=c_value,
                        class_weight="balanced",
                        solver="liblinear",
                    ),
                ),
            ]
        )

    def fit(self, train_questions_df: pd.DataFrame) -> None:
        option_df = build_option_level_frame(train_questions_df, with_targets=True)
        self.pipeline.fit(option_df["feature_text"], option_df["target"])

    def predict(self, questions_df: pd.DataFrame) -> Dict[str, List[str]]:
        predictions: Dict[str, List[str]] = {}
        option_df = build_option_level_frame(questions_df, with_targets=False)
        probs = self.pipeline.predict_proba(option_df["feature_text"])[:, 1]
        option_df = option_df.copy()
        option_df["proba"] = probs

        for qid, group in option_df.groupby("id", sort=False):
            best = group.sort_values(["proba", "label"], ascending=[False, True]).iloc[0]
            predictions[qid] = [str(best["label"])]
        return predictions

    def save(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline}, model_path)

    @classmethod
    def load(cls, model_path: str | Path) -> "OptionPairClassifier":
        model = cls()
        payload = joblib.load(model_path)
        model.pipeline = payload["pipeline"]
        return model


class _OptionPairTextDataset:
    def __init__(
        self,
        question_texts: List[str],
        option_texts: List[str],
        labels: Optional[List[int]] = None,
    ) -> None:
        self.question_texts = question_texts
        self.option_texts = option_texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.question_texts)

    def __getitem__(self, idx: int) -> Tuple[str, str, Optional[int]]:
        if self.labels is None:
            return self.question_texts[idx], self.option_texts[idx], None
        return self.question_texts[idx], self.option_texts[idx], self.labels[idx]


class TransformerOptionPairClassifier:
    """Cross-encoder baseline: Transformer binary classification on (question, option)."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 256,
        learning_rate: float = 2e-5,
        num_train_epochs: int = 2,
        batch_size: int = 16,
        weight_decay: float = 0.01,
        grad_clip_norm: float = 1.0,
        seed: int = 42,
        hf_cache_dir: str = "data/raw/hf_model_cache",
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.seed = seed
        self.hf_cache_dir = hf_cache_dir

        self.model = None
        self.tokenizer = None
        self.device = None

    def _ensure_initialized(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Transformer dependencies are missing. Install `torch` and `transformers`."
            ) from exc

        cache_dir = Path(self.hf_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            cache_dir=cache_dir,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, train_questions_df: pd.DataFrame) -> None:
        try:
            import torch
            from torch.optim import AdamW
            from torch.utils.data import DataLoader
            from transformers import get_linear_schedule_with_warmup
        except Exception as exc:
            raise RuntimeError(
                "Transformer dependencies are missing. Install `torch` and `transformers`."
            ) from exc

        self._ensure_initialized()
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        option_df = build_option_level_frame(train_questions_df, with_targets=True)
        dataset = _OptionPairTextDataset(
            question_texts=option_df["question"].astype(str).tolist(),
            option_texts=option_df["option_text"].astype(str).tolist(),
            labels=option_df["target"].astype(int).tolist(),
        )

        def collate_fn(batch):
            questions = [item[0] for item in batch]
            options = [item[1] for item in batch]
            labels = [item[2] for item in batch]
            encoded = self.tokenizer(
                questions,
                options,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded["labels"] = torch.tensor(labels, dtype=torch.long)
            return encoded

        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        total_steps = max(1, len(train_loader) * self.num_train_epochs)
        warmup_steps = max(0, int(0.1 * total_steps))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.model.train()
        for epoch in range(self.num_train_epochs):
            running_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Transformer epoch {epoch+1}/{self.num_train_epochs}"):
                labels = batch.pop("labels").to(self.device)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                running_loss += float(loss.detach().item())
            avg_loss = running_loss / max(1, len(train_loader))
            print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    def _predict_option_probabilities(self, questions_df: pd.DataFrame) -> pd.DataFrame:
        try:
            import torch
            from torch.utils.data import DataLoader
        except Exception as exc:
            raise RuntimeError(
                "Transformer dependencies are missing. Install `torch` and `transformers`."
            ) from exc

        self._ensure_initialized()
        option_df = build_option_level_frame(questions_df, with_targets=False).copy()
        dataset = _OptionPairTextDataset(
            question_texts=option_df["question"].astype(str).tolist(),
            option_texts=option_df["option_text"].astype(str).tolist(),
            labels=None,
        )

        def collate_fn(batch):
            questions = [item[0] for item in batch]
            options = [item[1] for item in batch]
            return self.tokenizer(
                questions,
                options,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

        probs: List[float] = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Transformer inference"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits
                positive_prob = torch.softmax(logits, dim=-1)[:, 1]
                probs.extend(positive_prob.detach().cpu().tolist())

        option_df["proba"] = probs
        return option_df

    def predict(self, questions_df: pd.DataFrame) -> Dict[str, List[str]]:
        predictions: Dict[str, List[str]] = {}
        option_df = self._predict_option_probabilities(questions_df)

        for qid, group in option_df.groupby("id", sort=False):
            best = group.sort_values(["proba", "label"], ascending=[False, True]).iloc[0]
            predictions[qid] = [str(best["label"])]
        return predictions

    def save(self, model_dir: str | Path) -> None:
        self._ensure_initialized()
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

        config = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
            "seed": self.seed,
            "hf_cache_dir": self.hf_cache_dir,
        }
        with (model_dir / "training_config.json").open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, ensure_ascii=True)

    @classmethod
    def load(cls, model_dir: str | Path) -> "TransformerOptionPairClassifier":
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Transformer dependencies are missing. Install `torch` and `transformers`."
            ) from exc

        model_dir = Path(model_dir)
        config_path = model_dir / "training_config.json"
        kwargs = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as fh:
                kwargs = json.load(fh)
        instance = cls(**kwargs)
        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance.model.to(instance.device)
        return instance
