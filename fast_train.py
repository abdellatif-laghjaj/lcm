import os
import sys

if __name__ == "__main__":
    """
    A convenience script to run training with optimized settings for faster iterations.
    This uses a smaller dataset, gradient accumulation, mixed precision, and other optimizations.
    """
    # Use a small subset of data for quick training/testing
    cmd = [
        "python", "train.py",
        "--model_type", "diffusion",
        "--diffusion_steps", "10",
        "--max_train_samples", "5000",  # Use only 5000 samples for quick iterations
        "--max_val_samples", "500",     # Use only 500 validation samples
        "--batch_size", "4",            # Small batch size to fit in memory
        "--gradient_accumulation_steps", "8",  # Effective batch size of 32
        "--fp16",                       # Use mixed precision training
        "--epochs", "3",                # Run fewer epochs for testing
        "--checkpoint_interval", "10",  # Save checkpoints every 10 minutes
        "--warmup_steps", "100",        # Shorter warmup
    ]
    
    # Add any additional arguments from command line
    cmd.extend(sys.argv[1:])
    
    # Print the command being run
    print("Running command:", " ".join(cmd))
    
    # Execute the command
    os.system(" ".join(cmd))