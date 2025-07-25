import json
import torch
from datasets import Dataset
from transformers import (
    TrainingArguments, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import os
from datetime import datetime

def load_training_data(file_path="mi_alpaca_format.json"):
    """Load the MI training data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} training examples from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Training data file {file_path} not found!")
        return []
    except Exception as e:
        print(f"Error loading training data: {e}")
        return []

def format_training_example(example):
    """Format a single training example into a conversation format."""
    system_prompt = example.get('system_prompt', '')
    question = example.get('question', '')
    response = example.get('response', '')
    
    # Create a chat format
    formatted_text = f"<|system|>\n{system_prompt}\n<|user|>\n{question}\n<|assistant|>\n{response}<|end|>"
    
    return {"text": formatted_text}

def prepare_dataset(training_data):
    """Prepare the dataset for training."""
    # Format all examples
    formatted_data = [format_training_example(example) for example in training_data]
    
    # Create Hugging Face dataset
    dataset = Dataset.from_list(formatted_data)
    
    return dataset

def tokenize_function(examples, tokenizer, max_length=1024):
    """Tokenize the examples."""
    # Tokenize the text
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=max_length,
        return_overflowing_tokens=False,
    )
    
    # For causal language modeling, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def setup_model_and_tokenizer(model_name="microsoft/DialoGPT-medium"):
    """Setup the model and tokenizer with LoRA."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens if they don't exist
    special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    new_tokens = []
    for token in special_tokens:
        if token not in tokenizer.get_vocab():
            new_tokens.append(token)
    
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine device and dtype for M1 optimization
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # MPS works best with float32
        print("Using Apple Silicon MPS acceleration!")
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True
    )
    
    # Move model to appropriate device
    model = model.to(device)
    
    # Resize token embeddings if we added new tokens
    if new_tokens:
        model.resize_token_embeddings(len(tokenizer))
    
    # Setup LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Low rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"] if "gpt" in model_name.lower() else ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer

def main():
    """Main training function."""
    print("Starting MI Dialogue Fine-tuning...")
    
    # Check for M1/M2 MPS support
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ Using Apple Silicon MPS acceleration!")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU")
    else:
        device = "cpu"
        print("Using CPU (will be slow)")
    
    print(f"Device: {device}")
    
    # Load training data
    training_data = load_training_data()
    if not training_data:
        print("No training data found. Exiting...")
        return
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(training_data)
    print(f"Dataset size: {len(dataset)}")
    
    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    model_name = "microsoft/DialoGPT-medium"  # Good for dialogue, works on Mac
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split dataset
    train_size = int(0.9 * len(tokenized_dataset))
    eval_size = len(tokenized_dataset) - train_size
    
    if eval_size > 0:
        split_dataset = tokenized_dataset.train_test_split(
            train_size=train_size,
            test_size=eval_size,
            seed=42
        )
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None
    
    print(f"Train size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval size: {len(eval_dataset)}")
    
    # Setup training arguments - M1 optimized
    output_dir = f"./mi_model_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # M1 optimized settings
    if torch.backends.mps.is_available():
        batch_size = 4  # M1 can handle slightly larger batches
        gradient_acc_steps = 4
        fp16_enabled = False  # MPS doesn't support fp16
        dataloader_workers = 0  # Avoid multiprocessing issues
    else:
        batch_size = 2
        gradient_acc_steps = 8
        fp16_enabled = False
        dataloader_workers = 0
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_acc_steps,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_steps=500 if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",  # Changed from evaluation_strategy
        save_strategy="steps",
        load_best_model_at_end=True if eval_dataset else False,
        learning_rate=2e-4,
        fp16=fp16_enabled,
        dataloader_num_workers=dataloader_workers,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb
        dataloader_pin_memory=False,  # Better for MPS
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting training...")
    try:
        trainer.train()
        
        # Save the final model
        print("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        print(f"Training completed! Model saved to {output_dir}")
        
        # Save training info
        info = {
            "model_name": model_name,
            "training_examples": len(training_data),
            "output_dir": output_dir,
            "completion_time": datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, "training_info.json"), "w") as f:
            json.dump(info, f, indent=2)
            
    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()