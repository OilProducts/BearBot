from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use the GPT-2 tokenizer as a base
# data = load_dataset("wikipedia", language="en", date="20230301", beam_runner='DirectRunner', split="train[:10%]")
tokenizer.pad_token = tokenizer.eos_token


data = load_dataset("wikipedia", "20220301.en")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

tokenized_data = data.map(tokenize_function, batched=True, batch_size=4096, num_proc=10, remove_columns=["text"])
tokenized_data.set_format("torch", columns=["input_ids", "attention_mask"])

tokenized_data.save_to_disk(f"tokenized_data")

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,
    n_ctx=512,
    n_embd=256,
    n_layer=8,
    n_head=8,
)

model = GPT2LMHeadModel(config)


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="path/to/output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=1000,
    logging_dir="path/to/logging",
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
)

trainer.train()

prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(input_ids, max_length=100, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)