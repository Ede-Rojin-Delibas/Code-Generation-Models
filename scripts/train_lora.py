import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, #It's for code generation model
                            AutoTokenizer, #text -> token
                            TrainingArguments, 
                            Trainer, #training loop
                            DataCollatorForLanguageModeling) #bathching
from peft import LoraConfig, get_peft_model #for adding LoRA adapters
from transformers import default_data_collator


MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH="data/processed/train.jsonl"
OUTPUT_DIR="outputs/lora_tinyllama"

MAX_LENGTH=256
BATCH_SIZE=1
EPOCHS=3
LR=2e-4

tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token=tokenizer.eos_token

dataset=load_dataset(
    "json",
    data_files=DATASET_PATH
)["train"]

dataset=dataset.shuffle(seed=42)
split = dataset.train_test_split(test_size=0.05)
train_dataset = split["train"]
eval_dataset = split["test"]

def tokenize_function(example):
    text=example["prompt"]+example["completion"]
    tokens=tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

    tokens["labels"]=tokens["input_ids"].copy()
    return tokens

tokenized_dataset=dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names
)

model=AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
lora_config=LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model=get_peft_model(model,lora_config)
model.print_trainable_parameters()

training_args=TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_steps=2,
    logging_steps=1,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
