# Qwen Researcher: QLoRA Fine-Tuning Tutorial

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Jalpan04/qwen-researcher)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Model_Weights-orange)](https://huggingface.co/jalpan04/qwen-researcher)

This document serves as your complete guide and documentation for fine-tuning the Qwen2.5-0.5B-Instruct model on the arXiv Computer Science dataset.

## Phase 1: Data Preparation (prepare_data.py)

The first script's job is to:
1. Extract: Pull the raw metadata from the arXiv archive.
2. Stream: Process the massive file line-by-line to save RAM.
3. Filter: Isolate papers in the cs. (Computer Science) category.
4. Format: Convert them into the ChatML structure for instruction tuning.

## Phase 2: The Training Script (train.py)

We use QLoRA (Quantized Low-Rank Adaptation) to train a 500M parameter model on a consumer GPU (RTX 4060).

### Key Concepts
*   4-bit Quantization: Loading the model in a compressed state to save VRAM.
*   LoRA Adapters: Training only a tiny "adapter" layer (Rank=16) instead of all weights.
*   Gradient Checkpointing: A memory-saving trick that calculates parts of the model on-the-fly.

## Phase 3: Merging & Conversion (merge_model.py)

Once training is done, we have Adapters. To use them in Ollama:
1. Merge: Combine the trained adapters back into the original Qwen model.
2. Convert: Use convert_hf_to_gguf.py to generate the final .gguf file.

## Troubleshooting & Optimization (Crucial for Windows)

During this project, we resolved several hardware-specific issues that are common for modern NVIDIA cards (RTX 40-series):

### 1. BF16 vs FP16
For RTX 4060 and newer cards, we used `bf16=True` and `fp16=False`. Standard FP16 can cause "NaN" losses or runtime errors during the gradient clipping stage on these architectures.

### 2. Dependency Conflicts (PyArrow)
We pinned `pyarrow==19.0.0` and `numpy<2.0.0`. Newer versions of PyArrow on Windows 10/11 can cause silent interpreter crashes during dataset loading.

### 3. UnicodeDecodeError (Charmap Fix)
On Windows, some libraries try to load UTF-8 files using the default "charmap" encoding. If you see an error about `byte 0x81`, run this before training:
```powershell
$env:PYTHONUTF8=1
```

### 4. Connection Reset (WinError 10054)
If uploading to Hugging Face fails with a Connection Reset, it is often due to Python's security handshake being blocked by a local firewall.

## Quick Start (Ollama)

1. Download: Get qwen-resercher.gguf from Hugging Face.
2. Setup: Place the Modelfile in the same folder.
3. Run:
   ```powershell
   ollama create qwen-researcher -f Modelfile
   ollama run qwen-researcher
   ```

## Repository Structure
- train.py: The QLoRA training engine.
- prepare_data.py: Dataset cleaning and formatting.
- merge_model.py: Final weight merging.
- check_gpu.py: Hardware diagnostic script.

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
