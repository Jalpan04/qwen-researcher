import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def merge_and_save():
    """
    Script to merge trained LoRA adapters back into the base Qwen model.
    This creates a standalone model folder that can be converted to GGUF format.
    """
    
    base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path = "./qwen-resercher"
    output_path = "./qwen-resercher-final"

    print(f"Loading base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load base model in FP16 for merging precision
    # device_map="auto" is used to handle large models, though 0.5B fits easily in VRAM.
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # 1. Load the trained adapter onto the base model
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    # 2. Merge the weights
    # merge_and_unload() combines the LoRA matrices with the original weights.
    # This results in a standard Transformer model (no adapters needed).
    print("Merging weights... this creates the final unified model.")
    model = model.merge_and_unload()

    # 3. Save the final model and tokenizer
    # This folder will be the source for the 'convert_hf_to_gguf.py' script.
    print(f"Saving unified model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("Merge complete.")

if __name__ == "__main__":
    merge_and_save()
