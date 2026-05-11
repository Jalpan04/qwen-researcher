import torch
import sys
import os

def check_gpu():
    print("-" * 30)
    print("PYTHON & TORCH CHECK")
    print("-" * 30)
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    print("\n" + "-" * 30)
    print("HARDWARE ACCELERATION")
    print("-" * 30)
    
    # Check for NVIDIA CUDA
    cuda_available = torch.cuda.is_available()
    print(f"NVIDIA CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
    else:
        print("Suggestion: If you have an NVIDIA GPU, you may need to install the CUDA-enabled version of PyTorch.")

    # Check for AMD/Intel (DirectML) - Requires 'torch-directml' package
    try:
        import torch_directml
        dml_available = True
        print(f"\nDirectML (AMD/Intel) Available: {dml_available}")
    except ImportError:
        print("\nDirectML (torch-directml) not installed.")

    print("\n" + "-" * 30)
    print("DIAGNOSTICS")
    print("-" * 30)
    if not cuda_available:
        print("If you have an NVIDIA GPU, try running:")
        print("nvidia-smi")
        print("\nTo see if the system sees your card at all.")

if __name__ == "__main__":
    check_gpu()
