import torch
import sys

def check_environment():
    """
    Diagnostic script to verify the GPU, CUDA, and PyTorch environment.
    Essential for troubleshooting fine-tuning setups on Windows.
    """
    
    print("--- System Diagnostics ---")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # 1. Check if CUDA (NVIDIA GPU support) is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # 2. Get GPU Details
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"Number of GPUs: {device_count}")
        print(f"Current Device Index: {current_device}")
        print(f"GPU Model: {device_name}")
        
        # 3. Memory Diagnostics
        # This helps check if other apps (like Chrome or Games) are eating VRAM
        total_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(current_device) / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(current_device) / (1024**3)
        free_memory = total_memory - reserved_memory
        
        print(f"Total VRAM: {total_memory:.2f} GB")
        print(f"Currently Reserved: {reserved_memory:.2f} GB")
        print(f"Currently Allocated: {allocated_memory:.2f} GB")
        print(f"Estimated Free VRAM: {free_memory:.2f} GB")
    else:
        print("WARNING: No NVIDIA GPU detected. Training will be extremely slow on CPU.")

if __name__ == "__main__":
    check_environment()
