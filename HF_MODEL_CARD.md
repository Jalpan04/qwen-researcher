---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
- qwen
- qwen2
- gguf
- f16
- computer-science
- research
- instruction-tuning
model_creator: jalpan04
model_name: Qwen Research Assistant
pipeline_tag: text-generation
quantized_by: f16
---

# Qwen Researcher (0.5B - GGUF)

A specialized version of **Qwen2.5-0.5B-Instruct** fine-tuned for Computer Science research assistance. This model has been instruction-tuned on a curated subset of 2,000 arXiv Computer Science papers to provide academic summaries and technical insights.

## Model Details

- **Developed by:** Jalpan04
- **Model type:** Causal Language Model
- **Language(s):** English
- **License:** Apache-2.0
- **Fine-tuned from model:** [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Training Method:** QLoRA (Rank 16, Alpha 32)
- **Dataset:** arXiv Computer Science Metadata (Instruction-formatted)

## Usage

### Local Deployment (Ollama)

1. Create a `Modelfile`:
```dockerfile
FROM ./qwen-resercher.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .User }}<|im_start|>user
{{ .User }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Output }}<|im_end|>"""
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
SYSTEM "You are a professional computer science researcher. Provide academic, detailed information based on research abstracts."
```

2. Run in terminal:
```bash
ollama create qwen-researcher -f Modelfile
ollama run qwen-researcher
```

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="jalpan04/qwen-researcher",
    filename="qwen-resercher.gguf",
)

response = llm.create_chat_completion(
    messages = [
        {"role": "system", "content": "You are a CS researcher."},
        {"role": "user", "content": "Summarize the latest trends in Neural Program Synthesis."}
    ]
)
print(response["choices"][0]["message"]["content"])
```

## Training Procedure

The model was trained on an **NVIDIA RTX 4060 (8GB)** using the following setup:
- **Optimization:** 4-bit NormalFloat (nf4) quantization.
- **Precision:** BFloat16 for stability on 40-series hardware.
- **Learning Rate:** 2e-4.
- **Batch Size:** 1 with Gradient Accumulation (16 steps).
- **Sequence Length:** 1024 tokens.

## Limitations

As a 0.5B parameter model, this is highly efficient but may exhibit hallucinations compared to larger models (7B+). It is best used for summarization and quick technical lookups rather than complex logical reasoning.
