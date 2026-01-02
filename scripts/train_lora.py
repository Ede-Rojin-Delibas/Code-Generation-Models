import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, #It's for code generation model
                            AutoTokenizer, #text -> token
                            TrainingArguments, 
                            Trainer, #training loop
                            DataCollatorForLanguageModeling) #bathching
from peft import LoraConfig, get_peft_model #for adding LoRA adapters

MODEL_NAME="microsoft/phi-2"
DATASET_PATH="data/processed/test_small.jsonl"
OUTPUT_DIR="outputs/phi2-lora"

MAX_LENGTH=512
BATCH_SIZE=2
EPOCHS=3
LR=2e-4

tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token=tokenizer.eos_token

dataset=load_dataset(
    "json",
    data_files=DATASET_PATH
)
dataset=dataset["train"]

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

lora_config=LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model=get_peft_model(model,lora_config)
model.print_trainable_parameters()

training_args=TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=1,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
