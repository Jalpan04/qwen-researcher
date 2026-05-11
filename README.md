# Ministral-3b QLoRA Fine-Tuning Tutorial & Documentation

This document serves as your complete guide and personal documentation for fine-tuning the `mistralai/Ministral-3b-instruct` model on the Kaggle arXiv dataset.

## Phase 1: Data Preparation (`prepare_data.py`)

The first script's job is to:
1. Extract the massive JSON snapshot from the ZIP file.
2. Stream through it (to avoid loading the whole 30GB+ file into RAM at once).
3. Filter out any paper that doesn't have `cs.` in its category.
4. Randomly select exactly 2,000 of those papers.
5. Format them into the standard OpenAI/ChatML structure required for modern instruction tuning.

The format we are converting the data to looks like this:
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert post-doctoral computer science researcher..."},
    {"role": "user", "content": "Can you summarize the research paper titled [Title]?"},
    {"role": "assistant", "content": "[Abstract]"}
  ]
}
```

## Phase 2: The Training Script (`train.py`)

The training script uses the Hugging Face ecosystem to fine-tune the model without requiring massive server farms. It utilizes `trl` (Transformer Reinforcement Learning) and `peft` to handle the heavy lifting.

---

## Under the Hood: Educational Concepts

To truly understand what you are doing, let's break down the mechanics of the code:

### 1. Why use 4-bit Quantization (QLoRA)?
A 3 Billion parameter model in standard 16-bit precision requires about 6GB of VRAM just to load, and double that to train. 
**Quantization** squashes the model's weights into smaller data types. By compressing the model to 4-bit precision, we shrink the memory footprint to roughly 1.5GB to 2GB. 
**QLoRA** (Quantized LoRA) is the technique where we load the base model frozen in 4-bit, but train a tiny set of adapter weights on top of it in 16-bit. This allows us to fine-tune large models on consumer GPUs (like RTX 3060/4090) without running out of memory.

### 2. What are PEFT and LoRA?
* **PEFT (Parameter-Efficient Fine-Tuning):** A broad category of techniques designed to fine-tune large models without updating every single parameter.
* **LoRA (Low-Rank Adaptation):** A specific PEFT technique. Instead of changing the original 3 Billion weights of the Mistral model, LoRA freezes them and injects small, trainable "matrix decompositions" into the neural network layers. When the model runs, it computes the frozen weights + the small LoRA weights. This reduces the number of trainable parameters by ~99% while maintaining near-full-parameter performance.

### 3. What do the LoRA `r` (rank) and `alpha` parameters control?
When we inject those tiny trainable matrices, we have to decide how "wide" they are.
* **`r` (Rank):** This is the "thickness" or dimension of the LoRA matrices. A higher rank (e.g., 32 or 64) means the adapter can learn more complex patterns, but uses more VRAM and takes longer to train. A lower rank (e.g., 8 or 16) is highly efficient. `r=16` is a great starting point.
* **`alpha`:** This is the scaling factor. It determines how "loud" or "impactful" the LoRA weights are compared to the frozen base model weights. The standard rule of thumb in the machine learning community is to set **`alpha = 2 * r`** (so if r=16, alpha=32).

### 4. What do `learning_rate` and `batch_size` dictate?
* **`learning_rate` (e.g., 2e-4):** This controls the "step size" the model takes when updating its weights. If it's too high, the model overshoots and learns nothing (loss explodes). If it's too low, it learns too slowly. Because LoRA only trains a tiny fraction of parameters, we typically use a *higher* learning rate (like `2e-4` or `1e-4`) compared to full fine-tuning (which often uses `2e-5`).
* **`batch_size`:** The number of papers the model looks at simultaneously before updating its weights. High batch sizes are more stable but eat VRAM. 
  * *The Trick:* We use a small `per_device_train_batch_size` (like 4) to fit in VRAM, but we use `gradient_accumulation_steps` (like 4) to mimic a batch size of 16. The model will calculate gradients for 4 batches of 4, add them up, and then make one big update.
