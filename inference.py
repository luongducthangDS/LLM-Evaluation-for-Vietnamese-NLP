import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel

# ======================
# CONFIG
# ======================
BASE_MODEL_NAME = "NlpHUST/gpt2-vietnamese"
ADAPTER_PATH = "lora_gpt2_viquad" # Thư mục bạn đã lưu model sau khi train
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Đang chạy trên thiết bị: {DEVICE}")

# ======================
# 1. LOAD TOKENIZER
# ======================
# Lưu ý: Load tokenizer từ thư mục đã train xong để giữ đúng config (pad_token, vocab_size)
print("Loading tokenizer...")
try:
    tokenizer = GPT2Tokenizer.from_pretrained(ADAPTER_PATH)
except:
    # Fallback nếu không tìm thấy tokenizer trong thư mục output
    print("⚠️ Không tìm thấy tokenizer đã lưu, load từ base model...")
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

# ======================
# 2. LOAD BASE MODEL & RESIZE
# ======================
print("Loading base model...")
base_model = GPT2LMHeadModel.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# QUAN TRỌNG: Phải resize lại embeddings của base model cho khớp với tokenizer
# Nếu bỏ qua bước này, khi load adapter sẽ bị lỗi lệch kích thước tensor
base_model.resize_token_embeddings(len(tokenizer))

# ======================
# 3. LOAD LORA ADAPTER
# ======================
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# Chuyển model sang chế độ đánh giá (không train nữa)
model.eval()
if DEVICE == "cuda":
    model.to(DEVICE)

print("✅ Model đã sẵn sàng!")

# ======================
# HÀM SINH VĂN BẢN
# ======================
def generate_answer(context, question):
    # Format prompt y hệt như lúc train
    prompt = (
        f"Ngữ cảnh:\n{context}\n\n"
        f"Câu hỏi: {question}\n"
        f"Trả lời:"
    )

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,      # Độ dài tối đa của câu trả lời
            do_sample=True,          # Random sampling để câu văn tự nhiên hơn
            top_p=0.9,               # Lấy các từ có xác suất cộng dồn 90%
            temperature=0.7,         # Độ sáng tạo (thấp = chính xác, cao = sáng tạo)
            repetition_penalty=1.2,  # Tránh lặp từ
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode kết quả
    # Chỉ lấy phần mới sinh ra (bỏ phần prompt ban đầu)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Cắt bỏ phần prompt để chỉ lấy câu trả lời
    answer = generated_text.replace(prompt, "").strip()

    return answer

# ======================
# TEST THỬ
# ======================
context_text = """
Trường Đại học Công nghệ Thông tin (UIT) là một trường đại học công lập tại Việt Nam, trực thuộc Đại học Quốc gia Thành phố Hồ Chí Minh.
Trường được thành lập vào ngày 8 tháng 6 năm 2006. UIT chuyên đào tạo về công nghệ thông tin và truyền thông.
"""

question_text = "Trường Đại học Công nghệ Thông tin được thành lập khi nào?"

print("\n" + "="*30)
print("CONTEXT:", context_text)
print("QUESTION:", question_text)
print("="*30)

answer = generate_answer(context_text, question_text)

print(f"\n🤖 MODEL TRẢ LỜI:\n{answer}")