print(">>> DEBUG: Starting script execution...")
print(">>> DEBUG: Importing libraries (transformers can take 10-20 seconds to initialize)...")
import torch
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
    print(">>> DEBUG: Entering train() function...")
    # Clear any leftover memory from the previous crash
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model_id = "Qwen/Qwen2.5-0.5B-Instruct" # A tiny 0.5B parameter model (under 1GB download)
    data_file = "arxiv_cs_2000.jsonl"
    output_dir = "./qwen-resercher-checkpoints"

    print(f">>> DEBUG: Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Setting padding side to right is often required for causal LMs during training
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(">>> DEBUG: Checking for GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> DEBUG: Using device: {device}")

    bnb_config = None
    if device == "cuda":
        print(">>> DEBUG: Configuring 4-bit Quantization (QLoRA) for GPU...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        print(">>> DEBUG: No GPU found. Skipping 4-bit quantization to improve CPU performance.")

    print(f">>> DEBUG: Loading Model from Hugging Face...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device
    )
    
    print(">>> DEBUG: Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    print(">>> DEBUG: Configuring LoRA adapters...")
    lora_config = LoraConfig(
        r=16, # Rank
        lora_alpha=32, # Alpha multiplier
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    print(">>> DEBUG: LoRA config created (SFTTrainer will inject the adapters).")

    print(f">>> DEBUG: Loading local dataset from {data_file}...")
    dataset = load_dataset("json", data_files=data_file, split="train")

    print(">>> DEBUG: Applying chat template to dataset...")

    # The SFTTrainer needs the messages transformed into the exact string format 
    # the model expects. We use the tokenizer's chat template.
    def format_chat_template(example):
        example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return example

    dataset = dataset.map(format_chat_template)

    print("6. Setting up SFT Configuration...")
    sft_config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_length=1024,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        logging_steps=1,
        max_steps=200,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        bf16=(device == "cuda"),
        fp16=False,
        run_name="qwen-cs-arxiv-finetune",
        report_to="none"
    )

    print("7. Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    print("8. Starting Training...")
    trainer.train()

    print("9. Saving the Adapter Model...")
    final_save_path = "qwen-resercher"
    trainer.model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"Training Complete! Adapter saved to {final_save_path}")

if __name__ == "__main__":
    train()
