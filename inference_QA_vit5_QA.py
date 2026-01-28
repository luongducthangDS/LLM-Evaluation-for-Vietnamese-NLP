# ============================================================
# Inference VietAI/vit5-base + LoRA (Question Answering)
# ============================================================

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ======================
# CONFIG
# ======================
BASE_MODEL_NAME = "VietAI/vit5-base"
ADAPTER_PATH = "lora_vit5_viquad"   # thư mục đã save sau khi finetune
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Đang chạy trên thiết bị: {DEVICE}")

# ======================
# 1. LOAD TOKENIZER
# ======================
print("Loading tokenizer...")
try:
    # Ưu tiên load tokenizer từ output_dir để giữ config nhất quán
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
except:
    print("⚠️ Không tìm thấy tokenizer trong thư mục adapter, load từ base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# ======================
# 2. LOAD BASE MODEL
# ======================
print("Loading base model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto" if DEVICE == "cuda" else None,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)

# ======================
# 3. LOAD LORA ADAPTER
# ======================
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

model.eval()
if DEVICE == "cuda":
    model.to(DEVICE)

print("✅ Model đã sẵn sàng inference!")

# ======================
# HÀM INFERENCE QA
# ======================
def generate_answer(context, question,
                    max_new_tokens=128,
                    num_beams=4):
    """
    Sinh câu trả lời cho bài toán QA theo đúng format lúc train
    """

    # Format input giống preprocess khi train
    input_text = (
        "trả lời câu hỏi dựa trên ngữ cảnh sau:\n\n"
        f"Ngữ cảnh: {context}\n\n"
        f"Câu hỏi: {question}"
    )

    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    # Generate (T5-style)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,      # beam search → ổn định, đúng ngữ nghĩa
            early_stopping=True,
            no_repeat_ngram_size=3,   # tránh lặp
        )

    # Decode output
    answer = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    ).strip()

    return answer

# ======================
# TEST THỬ
# ======================
context_text = """
Trường Đại học Công nghệ Thông tin (UIT) là một trường đại học công lập tại Việt Nam,
trực thuộc Đại học Quốc gia Thành phố Hồ Chí Minh.
Trường được thành lập vào ngày 8 tháng 6 năm 2006.
UIT chuyên đào tạo về công nghệ thông tin và truyền thông.
"""

question_text = "Trường Đại học Công nghệ Thông tin được thành lập khi nào?"

print("\n" + "=" * 40)
print("CONTEXT:\n", context_text)
print("QUESTION:", question_text)
print("=" * 40)

answer = generate_answer(context_text, question_text)

print("\n🤖 MODEL TRẢ LỜI:")
print(answer)
