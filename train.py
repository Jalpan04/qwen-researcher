import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def train():
    """
    Main training function to fine-tune Qwen2.5-0.5B using QLoRA.
    This script is optimized for consumer GPUs (like RTX 4060) and 
    addresses specific Windows/NVIDIA driver compatibility issues.
    """
    
    # Clear CUDA memory before starting to prevent fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    data_file = "arxiv_cs_2000.jsonl"
    output_dir = "./qwen-resercher-checkpoints"

    # 1. Load Tokenizer
    # We use the standard Qwen tokenizer. Padding is set to the right for Causal LM training.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. BitsAndBytes Configuration (QLoRA)
    # This allows us to load the model in 4-bit, saving ~75% of VRAM.
    # We use 'nf4' (Normal Float 4) which is more accurate than standard 4-bit.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 for RTX 40-series stability
    )

    # 3. Load Base Model
    # device_map="auto" automatically balances the model across available GPU/CPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare model for kbit training (adds gradient checkpointing and layer freezing)
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA Configuration
    # We only train a tiny 'adapter' (rank 16) instead of all 500M parameters.
    # target_modules defines which layers are adapted (Q, K, V, O in attention blocks).
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 5. Load Dataset
    # We load the ChatML-formatted data prepared by prepare_data.py
    dataset = load_dataset("json", data_files=data_file, split="train")

    # 6. Training Arguments
    # Optimized for 8GB VRAM cards.
    # bf16=True is essential for RTX 40-series to avoid precision errors.
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16, # Mimics a batch size of 16 for stability
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=200,
        save_steps=100,
        optim="paged_adamw_8bit", # Saves memory by offloading optimizer states
        bf16=True,
        fp16=False,
        dataset_text_field="text", # SFTTrainer uses this to find the input strings
        max_seq_length=1024,
        gradient_checkpointing=True,
        packing=False
    )

    # 7. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        tokenizer=tokenizer,
    )

    # 8. Start Training
    print("Starting fine-tuning...")
    trainer.train()

    # 9. Save the Adapter
    # This only saves the tiny adapter files (~50MB), not the full model.
    print("Saving fine-tuned adapter...")
    trainer.model.save_pretrained("qwen-resercher")
    tokenizer.save_pretrained("qwen-resercher")
    print("Training complete.")

if __name__ == "__main__":
    train()
