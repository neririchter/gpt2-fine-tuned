from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import torch

# Load GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the padding token to the EOS token (GPT-2 does not have a specific pad token)
tokenizer.pad_token = tokenizer.eos_token

# Define the LoRA configuration with correct target layers
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],  # Corrected target modules for GPT-2
    lora_dropout=0.1,
)

# Apply LoRA to the GPT-2 model
model = get_peft_model(model, lora_config)

# Load a small dataset (you can adjust this as needed)
dataset = load_dataset("imdb")
train_dataset = dataset["train"].shuffle(seed=42).select(range(100))  # Select a small subset
test_dataset = dataset["test"].shuffle(seed=42).select(range(100))  # Select a small subset


# Tokenize the dataset
def preprocess(examples):
    # We use the labels for language modeling (shift by 1)
    inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    labels = inputs["input_ids"].copy()

    # Shift the input to make the labels
    labels = [-100 if token == tokenizer.pad_token_id else token for token in labels]  # Ignore pad tokens
    inputs["labels"] = labels  # Set labels in the tokenizer output
    return inputs


# Apply preprocessing
encoded_train_dataset = train_dataset.map(preprocess, batched=True)
encoded_test_dataset = test_dataset.map(preprocess, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_finetune_results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_test_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_gpt2_model")
tokenizer.save_pretrained("./fine_tuned_gpt2_model")

print("Fine-tuned GPT-2 model and tokenizer saved.")
