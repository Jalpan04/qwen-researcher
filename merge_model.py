import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def merge():
    base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path = "./qwen-resercher"
    output_path = "./qwen-resercher-final"

    print(f">>> Loading base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load base model in FP16 for merging
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f">>> Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print(">>> Merging weights... (this might take a minute)")
    model = model.merge_and_unload()

    print(f">>> Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(">>> DONE! Your merged model is ready in 'qwen-resercher-final'")

if __name__ == "__main__":
    merge()
