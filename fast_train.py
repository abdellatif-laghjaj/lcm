import os
import sys
import torch

if __name__ == "__main__":
    """
    A convenience script to run training with optimized settings for faster iterations.
    This uses a smaller dataset, gradient accumulation, and other optimizations.
    """
    # Check if CUDA is available and set options accordingly
    cuda_available = torch.cuda.is_available()

    # Base command with settings that work on any machine
    cmd = [
        "python",
        "train.py",
        "--model_type",
        "diffusion",
        "--diffusion_steps",
        "10",
        "--max_train_samples",
        "5000",  # Use only 5000 samples for quick iterations
        "--max_val_samples",
        "500",  # Use only 500 validation samples
        "--batch_size",
        "4",  # Small batch size to fit in memory
        "--gradient_accumulation_steps",
        "8",  # Effective batch size of 32
        "--epochs",
        "3",  # Run fewer epochs for testing
        "--checkpoint_interval",
        "10",  # Save checkpoints every 10 minutes
        "--warmup_steps",
        "100",  # Shorter warmup
        # Disable wandb by default to avoid prompts
        "--wandb_mode",
        "disabled",  # Disable wandb by default
    ]

    # Add CUDA-specific optimizations only if available
    if cuda_available:
        cmd.extend(
            [
                "--fp16",  # Only use mixed precision with CUDA
                "--pin_memory",  # Only use pin_memory with CUDA
                "--num_workers",
                "2",  # Use dataloader workers with CUDA
            ]
        )
    else:
        print("CUDA is not available, running in CPU-only mode (slower)")
        # Reduce batch size further for CPU-only mode
        cmd.extend(["--batch_size", "2", "--gradient_accumulation_steps", "16"])

    # Add any additional arguments from command line
    cmd.extend(sys.argv[1:])

    # Print the command being run
    print("Running command:", " ".join(cmd))

    # Execute the command
    os.system(" ".join(cmd))
