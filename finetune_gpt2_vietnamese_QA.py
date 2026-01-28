# ============================================================
# Finetune GPT2-Vietnamese với LoRA - FIXED VOCAB SIZE
# ============================================================

import os
import numpy as np
import torch    
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from evaluate import load as load_metric

# ======================
# CLEAR CUDA CACHE
# ======================
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("✅ CUDA cache cleared")

# ======================
# CONFIG
# ======================
MODEL_NAME = "NlpHUST/gpt2-vietnamese"
DATASET_NAME = "taidng/UIT-ViQuAD2.0"

MAX_LENGTH = 512
TRAIN_SAMPLES = 5000
EVAL_SAMPLES = 200
OUTPUT_DIR = "lora_gpt2_viquad"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ======================
# LOAD TOKENIZER & MODEL
# ======================
print("Loading tokenizer and model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# Load model
if torch.cuda.is_available():
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        load_in_8bit=True
    )
else:
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# ============================================
# FIX QUAN TRỌNG: Resize model embeddings
# ============================================
print(f"Model vocab size TRƯỚC: {model.config.vocab_size}")
print(f"Tokenizer vocab size: {len(tokenizer)}")

# Set pad token TRƯỚC KHI resize
tokenizer.pad_token = tokenizer.eos_token

# Resize model embeddings để khớp với tokenizer
model.resize_token_embeddings(len(tokenizer))

print(f"Model vocab size SAU resize: {model.config.vocab_size}")

# Update config
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id

# ======================
# APPLY LORA
# ======================
print("Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print(f"Model device: {next(model.parameters()).device}")

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

train_ds = dataset["train"].filter(has_answer).shuffle(seed=42).select(range(TRAIN_SAMPLES))
eval_ds = dataset["validation"].filter(has_answer).shuffle(seed=42).select(range(EVAL_SAMPLES))

print(f"Train samples: {len(train_ds)}")
print(f"Eval samples: {len(eval_ds)}")

# ======================
# PROMPT BUILDER
# ======================
def build_prompt(example):
    return (
        f"Ngữ cảnh:\n{example['context']}\n\n"
        f"Câu hỏi: {example['question']}\n"
        f"Trả lời:"
    )

# ======================
# PREPROCESS FUNCTION
# ======================
def preprocess(example):
    prompt = build_prompt(example)
    answer = example["answers"]["text"][0].strip()
    full_text = prompt + " " + answer

    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Tokenize prompt
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    )["input_ids"]

    prompt_len = len(prompt_ids)

    # Tạo labels - QUAN TRỌNG: Kiểm tra vocab size
    labels = []
    vocab_size = model.config.vocab_size

    for i, (token_id, mask) in enumerate(zip(input_ids, attention_mask)):
        if i < prompt_len:
            # Mask prompt
            labels.append(-100)
        elif mask == 0:
            # Mask padding
            labels.append(-100)
        elif 0 <= token_id < vocab_size:
            # Token hợp lệ
            labels.append(token_id)
        else:
            # Token không hợp lệ - KHÔNG BAO GIỜ XẢY RA NẾU ĐÃ RESIZE
            print(f"⚠️ Warning: token_id {token_id} >= vocab_size {vocab_size}")
            labels.append(-100)

    tokenized["labels"] = labels
    return tokenized

print("Preprocessing datasets...")
train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
eval_ds = eval_ds.map(preprocess, remove_columns=eval_ds.column_names)

print("✅ Preprocessing completed!")

# ======================
# METRIC (ROUGE-L)
# ======================
rouge = load_metric("rouge")

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Logits lúc này thường là một tuple hoặc ndarray cực lớn
    # Chúng ta lấy argmax để lấy token ID, giúp giảm kích thước bộ nhớ hàng chục lần
    if isinstance(logits, tuple):
        logits = logits[0]

    # Chuyển sang ID dự đoán ngay lập tức để tiết kiệm RAM
    predictions = np.argmax(logits, axis=-1)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Xử lý labels (giữ nguyên logic của bạn nhưng tối ưu hơn)
    decoded_labels = []
    for label in labels:
        valid_tokens = [
            token for token in label
            if token != -100 and 0 <= token < tokenizer.vocab_size
        ]
        decoded_label = tokenizer.decode(valid_tokens, skip_special_tokens=True)
        decoded_labels.append(decoded_label)

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )
    return {
        "rougeL": result["rougeL"],
    }

# ======================
# TRAINING ARGS
# ======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,       # Giảm xuống 1 cho an toàn
    gradient_accumulation_steps=8,
    eval_accumulation_steps=1,          # QUAN TRỌNG: Giải phóng bộ nhớ liên tục
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=torch.cuda.is_available(),
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    logging_first_step=True,
    optim="adamw_torch",
)

# ======================
# TRAINER
# ======================
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# ======================
# TRAIN
# ======================
print("Starting training...")
try:
    trainer.train()
    print("✅ Training completed successfully!")
except Exception as e:
    print(f"❌ Training failed with error: {e}")
    import traceback
    traceback.print_exc()
    raise

# ======================
# SAVE MODEL
# ======================
print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Finetune hoàn tất, model đã được lưu tại:", OUTPUT_DIR)