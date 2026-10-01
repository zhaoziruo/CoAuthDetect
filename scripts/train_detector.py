#!/usr/bin/env python
# coding=utf-8
"""Train a 4-class human-AI co-authorship detector.

Reference implementation for the paper "Towards Fine-grained and Robust
Detection of Human-AI Co-authorship". Classifies a document by its *degree* of
AI involvement rather than as a binary human/AI decision:

    0 = human       original corpus text
    1 = low_ai      LLM rewrote it sentence by sentence, surface edits only
    2 = high_ai     LLM heavily rewrote it, may restructure
    3 = fully_ai    LLM wrote it from a short prefix

Input is the CSV written by ``scripts/build_splits.py`` -- two columns, ``text``
and ``label``. The release does not ship those CSVs or the human-written class;
build them first:

    python scripts/build_splits.py --source xsum --generator gpt4o \
        --human xsum_human.json --adversarial level1 --out-dir splits/

    python scripts/train_detector.py \
        --model_name_or_path FacebookAI/roberta-large \
        --do_train --do_eval --do_predict \
        --train_file splits/train.csv \
        --validation_file splits/val.csv \
        --test_file splits/test-level1.csv \
        --max_seq_length 512 \
        --per_device_train_batch_size 8 --gradient_accumulation_steps 2 \
        --learning_rate 3e-5 --num_train_epochs 5 --fp16 \
        --output_dir runs/xsum-gpt4o-level1

Accepts the full HuggingFace ``TrainingArguments`` surface in addition to the
arguments below.

Robustness evaluation: train once at one adversarial level, then run
``--do_predict`` against each of ``test-benign.csv``, ``test-level1.csv`` and
``test-level2.csv`` in turn. Those three conditions are the robustness table in
the paper; ``build_splits.py`` writes all three from a single invocation.

Metrics written to ``--output_dir``: accuracy, macro-F1, per-class F1, and
macro-AUC (one-vs-rest), in ``eval_results.json`` and ``test_results.json``,
with per-row class probabilities in ``predict_probabilities.csv``.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import transformers
from datasets import load_dataset
from scipy.special import softmax
from sklearn.metrics import f1_score, roc_auc_score
from torch import tensor
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

logger = logging.getLogger(__name__)

#: must match scripts/build_splits.py LABEL
LABEL_NAMES = {0: "human", 1: "low_ai", 2: "high_ai", 3: "fully_ai"}


@dataclass
class DataTrainingArguments:
    """Data files and preprocessing."""

    train_file: str = field(metadata={"help": "Training CSV from build_splits.py"})
    validation_file: str = field(metadata={"help": "Validation CSV from build_splits.py"})
    test_file: str = field(
        metadata={"help": "Test CSV -- one of test-benign / test-level1 / test-level2"}
    )
    text_column: str = field(default="text", metadata={"help": "Text column name."})
    label_column: str = field(default="label", metadata={"help": "Label column name."})
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Max tokens after tokenization. 512 for RoBERTa/BERT; "
                          "longformer-base-4096 accepts up to 4096."},
    )
    pad_to_max_length: bool = field(default=True, metadata={"help": "Pad to max_seq_length."})
    overwrite_cache: bool = field(default=False, metadata={"help": "Ignore the dataset cache."})
    class_weights: str = field(
        default="balanced",
        metadata={"help": "'balanced' = inverse class frequency. Adversarial training "
                          "levels add DIPPER rows to the three AI classes but not to the "
                          "human class, so training at level1/level2 is imbalanced and "
                          "wants weighting; a benign split is balanced and does not."},
    )

    def __post_init__(self):
        files = [self.train_file, self.validation_file, self.test_file]
        for path in files:
            if not path.endswith((".csv", ".json")):
                raise ValueError(f"{path} must be .csv or .json")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} not found. Build the splits first:\n"
                    f"  python scripts/build_splits.py --source <corpus> "
                    f"--generator <model> --human <file> --out-dir <dir>"
                )

        extensions = {f.rsplit(".", 1)[-1] for f in files}
        if len(extensions) > 1:
            raise ValueError(f"All data files need the same extension, got {extensions}")

        resolved = [os.path.realpath(f) for f in files]
        if len(set(resolved)) != 3:
            raise ValueError(
                "train / validation / test must be three different files.\n"
                f"  train: {self.train_file}\n  val:   {self.validation_file}\n"
                f"  test:  {self.test_file}"
            )

        if self.class_weights not in ("balanced", "none"):
            raise ValueError("--class_weights must be 'balanced' or 'none'")


@dataclass
class ModelArguments:
    """Model configuration."""

    model_name_or_path: str = field(
        metadata={"help": "HF model id or local path. The paper reports "
                          "FacebookAI/roberta-large and allenai/longformer-base-4096."}
    )
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    from_scratch: bool = field(
        default=False, metadata={"help": "Random init instead of pretrained weights."}
    )


class WeightedTrainer(Trainer):
    """Trainer with an optional per-class loss weight vector."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None if self.class_weights is None else self.class_weights.to(logits.device)
        loss = CrossEntropyLoss(weight=weight)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def build_metrics(num_labels: int):
    """compute_metrics closure: accuracy, macro-F1, per-class F1, macro-AUC.

    Macro-F1 is the headline metric: the classes are not equally easy -- high_ai
    and fully_ai are the confusable pair -- so accuracy alone hides where a
    model is actually failing. Per-class F1 is reported for the same reason.
    """
    is_binary = num_labels == 2
    classes = list(range(num_labels))

    def compute_metrics(p: EvalPrediction):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(logits, axis=1)
        labels = p.label_ids
        probs = softmax(logits, axis=1)

        result = {
            "accuracy": float((preds == labels).mean()),
            "macro_f1": float(f1_score(labels, preds, average="macro",
                                       labels=classes, zero_division=0)),
        }

        for i, score in enumerate(f1_score(labels, preds, average=None,
                                           labels=classes, zero_division=0)):
            result[f"f1_{LABEL_NAMES.get(i, i)}"] = float(score)

        try:
            if is_binary:
                result["auc"] = float(roc_auc_score(labels, probs[:, 1]))
            else:
                result["auc"] = float(roc_auc_score(
                    labels, probs, multi_class="ovr", average="macro", labels=classes
                ))
        except ValueError as exc:
            logger.warning(f"AUC unavailable: {exc}")
            result["auc"] = float("nan")

        return result

    return compute_metrics


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)

    # ---------------- data ----------------
    data_files = {
        "train": data_args.train_file,
        "validation": data_args.validation_file,
        "test": data_args.test_file,
    }
    extension = data_args.train_file.rsplit(".", 1)[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    logger.info(f"Loaded datasets: {raw_datasets}")

    label_list = sorted(raw_datasets["train"].unique(data_args.label_column))
    num_labels = len(label_list)
    if label_list != list(range(num_labels)):
        # Labels are positional, so a split missing a class would shift the
        # mapping relative to training and silently scramble every metric.
        raise ValueError(
            f"Expected contiguous integer labels 0..{num_labels - 1}, got {label_list}. "
            f"build_splits.py emits {LABEL_NAMES}."
        )
    logger.info(f"{num_labels} labels: { {i: LABEL_NAMES.get(i, i) for i in label_list} }")

    counts = raw_datasets["train"].to_pandas()[data_args.label_column].value_counts()
    total = len(raw_datasets["train"])
    logger.info(f"Train class counts: "
                f"{ {LABEL_NAMES.get(k, k): int(v) for k, v in counts.sort_index().items()} }")

    if data_args.class_weights == "balanced":
        class_weights = tensor(
            [total / (num_labels * counts[label]) for label in label_list]
        ).float()
        logger.info(f"Class weights (balanced): {class_weights.tolist()}")
    else:
        class_weights = None
        logger.info("Class weights: none (plain cross-entropy)")

    # ---------------- model ----------------
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    if model_args.from_scratch:
        logger.info("Random initialization (--from_scratch)")
        model = AutoModelForSequenceClassification.from_config(config)
    else:
        logger.info(f"Loading pretrained weights from {model_args.model_name_or_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            ignore_mismatched_sizes=True,
        )

    model.config.id2label = {i: LABEL_NAMES.get(i, str(i)) for i in label_list}
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}

    # ---------------- tokenize ----------------
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"--max_seq_length {data_args.max_seq_length} exceeds this tokenizer's limit "
            f"({tokenizer.model_max_length}); clamping."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess(examples):
        result = tokenizer(
            examples[data_args.text_column],
            padding=padding,
            max_length=max_seq_length,
            truncation=True,
        )
        result["label"] = examples[data_args.label_column]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        tokenized = raw_datasets.map(
            preprocess,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing",
        )

    train_dataset = tokenized["train"] if training_args.do_train else None
    eval_dataset = tokenized["validation"] if training_args.do_eval else None
    predict_dataset = tokenized["test"] if training_args.do_predict else None
    for name, dataset in (("train", train_dataset), ("validation", eval_dataset),
                          ("test", predict_dataset)):
        if dataset is not None:
            logger.info(f"{name} samples: {len(dataset)}")

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    compute_metrics = build_metrics(num_labels)
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        class_weights=class_weights,
    )

    # ---------------- train ----------------
    if training_args.do_train:
        logger.info("*** Training ***")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Model saved to {training_args.output_dir}")

    # ---------------- eval ----------------
    if training_args.do_eval:
        logger.info("*** Evaluation ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # ---------------- predict ----------------
    if training_args.do_predict:
        logger.info("*** Prediction ***")
        has_labels = data_args.label_column in raw_datasets["test"].column_names
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict")
        logits = predictions.predictions
        probs = softmax(logits, axis=1)
        pred_ids = np.argmax(logits, axis=1)

        np.savetxt(os.path.join(training_args.output_dir, "predict_probabilities.csv"),
                   probs, delimiter=",",
                   header=",".join(LABEL_NAMES[i] for i in label_list), comments="")

        with open(os.path.join(training_args.output_dir, "predictions.txt"), "w") as fh:
            fh.write("index\tprediction\tlabel_name\n")
            for idx, pred_id in enumerate(pred_ids):
                fh.write(f"{idx}\t{pred_id}\t{LABEL_NAMES.get(pred_id, pred_id)}\n")

        if has_labels:
            test_metrics = compute_metrics(
                EvalPrediction(
                    predictions=logits,
                    label_ids=np.array(raw_datasets["test"][data_args.label_column]),
                )
            )
            test_metrics["test_samples"] = len(predict_dataset)
            test_metrics["test_file"] = data_args.test_file
            with open(os.path.join(training_args.output_dir, "test_results.json"), "w") as fh:
                json.dump(test_metrics, fh, indent=2)
            logger.info("*** Test results ***")
            for key, value in test_metrics.items():
                logger.info(f"  {key} = {value}")
        else:
            logger.info("Test file has no label column; wrote predictions only.")

    logger.info("*** Complete ***")


if __name__ == "__main__":
    main()
