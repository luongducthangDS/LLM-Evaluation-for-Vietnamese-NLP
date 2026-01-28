# ============================================================
# Finetune VietAI/vit5-base với LoRA cho Question Answering
# Dataset: UIT-ViQuAD2.0
# ============================================================

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
from evaluate import load as load_metric

# ======================
# CLEAR CUDA
# ======================
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("✅ CUDA cache cleared")

# ======================
# CONFIG
# ======================
MODEL_NAME = "VietAI/vit5-base"
DATASET_NAME = "taidng/UIT-ViQuAD2.0"

MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 128

TRAIN_SAMPLES = 5000
EVAL_SAMPLES = 200

OUTPUT_DIR = "lora_vit5_viquad"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ======================
# LOAD TOKENIZER & MODEL
# ======================
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" if torch.cuda.is_available() else None,
    load_in_8bit=torch.cuda.is_available(),
)

# ======================
# APPLY LORA
# ======================
print("Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
    target_modules=["q", "k", "v", "o"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ======================
# LOAD DATASET
# ======================
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

def has_answer(example):
    return (
        example["is_impossible"] is False
        and len(example["answers"]["text"]) > 0
    )

train_ds = (
    dataset["train"]
    .filter(has_answer)
    .shuffle(seed=42)
    .select(range(TRAIN_SAMPLES))
)

eval_ds = (
    dataset["validation"]
    .filter(has_answer)
    .shuffle(seed=42)
    .select(range(EVAL_SAMPLES))
)

print(f"Train samples: {len(train_ds)}")
print(f"Eval samples: {len(eval_ds)}")

# ======================
# PREPROCESS
# ======================
def preprocess(example):
    # T5-style QA input
    source_text = (
        f"trả lời câu hỏi dựa trên ngữ cảnh sau:\n\n"
        f"Ngữ cảnh: {example['context']}\n\n"
        f"Câu hỏi: {example['question']}"
    )

    target_text = example["answers"]["text"][0].strip()

    model_inputs = tokenizer(
        source_text,
        max_length=MAX_SOURCE_LEN,
        truncation=True,
        padding="max_length",
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            target_text,
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding="max_length",
        )

    # Mask padding tokens trong labels
    labels["input_ids"] = [
        token if token != tokenizer.pad_token_id else -100
        for token in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Preprocessing datasets...")
train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
eval_ds = eval_ds.map(preprocess, remove_columns=eval_ds.column_names)
print("✅ Preprocessing completed!")

# ======================
# METRIC (ROUGE-L)
# ======================
rouge = load_metric("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(
        preds, skip_special_tokens=True
    )

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True
    )

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )

    return {"rougeL": result["rougeL"]}

# ======================
# DATA COLLATOR
# ======================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

# ======================
# TRAINING ARGS
# ======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=torch.cuda.is_available(),
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    optim="adamw_torch",
)

# ======================
# TRAINER
# ======================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ======================
# TRAIN
# ======================
print("Starting training...")
trainer.train()
print("✅ Training completed!")

# ======================
# SAVE MODEL
# ======================
print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Finetune hoàn tất, model đã được lưu tại:", OUTPUT_DIR)
