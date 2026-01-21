import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "/content/drive/MyDrive/code-llm-finetune/hf_cache/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/.no_exist/fe8a4ea1ffedaf415f4da2f062534de366a451e6/adapter_config.json"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

prompt = """### Task:
Write a Python code based on the following description.

Validate whether a given string is a valid email address.

### Answer:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2,
        top_p=0.9,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
