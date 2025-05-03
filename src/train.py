import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import wandb
import os
import time
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt
import spacy

from baselcm import BaseLCM, SonarEncoder
from utils import (
    GloveDataset,
    add_noise_to_embeddings,
    compute_metrics,
    plot_training_history,
    setup_data_processing_pipeline,
    split_train_test,
    save_config,
)

# Set random seed for reproducibility
torch.manual_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Training and evaluation for BaseLCM")

    # Model parameters
    parser.add_argument(
        "--input_dim", type=int, default=256, help="Input dimension for the model"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="Hidden dimension for the model"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of heads for the model"
    )
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of layers for the model"
    )
    parser.add_argument(
        "--ff_dim", type=int, default=2048, help="Feedforward dimension for the model"
    )
    parser.add_argument(
        "--output_dim", type=int, default=256, help="Output dimension for the model"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--sequence_length", type=int, default=10, help="Sequence length for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for training"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization factor)",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.05,
        help="Noise level for the target embeddings",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Fraction of data to use for testing",
    )
    parser.add_argument(
        "--early_stopping", type=int, default=5, help="Early stopping patience"
    )

    # Data parameters
    parser.add_argument(
        "--hf_data", type=str, default="oscar", help="Hugging Face dataset name"
    )
    parser.add_argument(
        "--hf_config",
        type=str,
        default="unshuffled_deduplicated_en",
        help="Configuration for the Hugging Face dataset",
    )
    parser.add_argument(
        "--text_column", type=str, default="text", help="Text column in the dataset"
    )
    parser.add_argument(
        "--lang", type=str, default="en", help="Language for the dataset"
    )
    parser.add_argument(
        "--data_sample",
        type=int,
        default=1000,
        help="Number of samples to use from the dataset",
    )

    # Other parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Use Weights and Biases for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="base-lcm",
        help="Weights and Biases project name",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_models",
        help="Directory to save models and results",
    )
    parser.add_argument(
        "--model_name", type=str, default="base_lcm_model", help="Name of the model"
    )
    parser.add_argument(
        "--plot_history", action="store_true", help="Plot training history"
    )

    return parser.parse_args()


# Centralized device management
def to_device(data, device):
    """Move data to specified device.

    Args:
        data: Data to move
        device: Device to move the data to

    Returns:
        Data on the specified device
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


def train_epoch(
    model, dataloader, target_embeddings, optimizer, criterion, device, batch_size
):
    """Train model for one epoch.

    Args:
        model: Model to train
        dataloader: DataLoader for training data
        target_embeddings: Target embeddings
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        batch_size: Batch size

    Returns:
        float: Average loss for the epoch
    """
    model.train()
    running_loss = 0.0

    # Wrapping the dataloader with tqdm for batch progress
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")

    for batch_idx, inputs in pbar:
        inputs = to_device(inputs, device)
        batch_targets = target_embeddings[
            batch_idx * batch_size : (batch_idx + 1) * batch_size
        ]
        if len(batch_targets) != inputs.size(0):
            continue  # Skip incomplete batches at the end

        optimizer.zero_grad()
        output_embeddings = model(inputs)
        loss = criterion(output_embeddings, batch_targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return running_loss / len(dataloader)


def evaluate(model, dataloader, target_embeddings, criterion, device, batch_size):
    """Evaluate model.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        target_embeddings: Target embeddings
        criterion: Loss function
        device: Device to use
        batch_size: Batch size

    Returns:
        tuple: Average loss and metrics dictionary
    """
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            inputs = to_device(inputs, device)
            batch_targets = target_embeddings[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]
            if len(batch_targets) != inputs.size(0):
                continue  # Skip incomplete batches at the end

            output_embeddings = model(inputs)
            loss = criterion(output_embeddings, batch_targets)
            running_loss += loss.item()

            all_outputs.append(output_embeddings.cpu())
            all_targets.append(batch_targets.cpu())

    # Concatenate all outputs and targets
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    metrics = compute_metrics(all_outputs, all_targets)

    return running_loss / len(dataloader), metrics


def train(args):
    """Main training function.

    Args:
        args: Command line arguments
    """
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(args.save_dir, f"{args.model_name}_config.json")
    save_config(vars(args), config_path)

    # Initialize Weights & Biases if specified
    if args.wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    # Initialize model
    model = BaseLCM(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        output_dim=args.output_dim,
    ).to(device)

    # Initialize encoder
    print("Initializing encoder...")
    encoder = SonarEncoder(device=device)

    # Load dataset
    print(f"Loading dataset: {args.hf_data}")
    try:
        if args.hf_config:
            df = load_dataset(args.hf_data, args.hf_config, split="train").select(
                range(args.data_sample)
            )
        else:
            df = load_dataset(args.hf_data, split="train").select(
                range(args.data_sample)
            )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using a fallback dataset...")
        df = load_dataset("oscar", "unshuffled_deduplicated_en", split="train").select(
            range(args.data_sample)
        )

    # Split text into sentences
    print("Splitting text into sentences...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        import subprocess

        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    # Function to split text into sentences
    def split_into_sentences(text):
        doc = nlp(text)
        return [sent.text for sent in doc.sents]

    processed_texts = []
    for text in tqdm(df[args.text_column], desc="Processing Texts"):
        sentences = split_into_sentences(text)
        processed_texts.extend(sentences)

    print(f"Number of sentences: {len(processed_texts)}")

    # Encode the processed sentences
    print("Encoding sentences...")
    input_embeddings, stats = setup_data_processing_pipeline(
        processed_texts, encoder, lang=args.lang, batch_size=args.batch_size
    )
    input_embeddings = input_embeddings.to(device)

    # Update model with statistics for normalization
    model.prenet.scaler_mean = stats["mean"].mean().item()
    model.prenet.scaler_std = stats["std"].mean().item()
    model.postnet.scaler_mean = stats["mean"].mean().item()
    model.postnet.scaler_std = stats["std"].mean().item()

    # Free up memory
    del encoder
    torch.cuda.empty_cache()

    # Split data into train and test sets
    train_embeddings, test_embeddings = split_train_test(
        input_embeddings, args.test_size
    )

    # Create datasets and dataloaders
    train_dataset = GloveDataset(
        train_embeddings, args.sequence_length, args.batch_size
    )
    test_dataset = GloveDataset(test_embeddings, args.sequence_length, args.batch_size)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Create target embeddings with noise
    train_targets = add_noise_to_embeddings(train_embeddings, args.noise_level).to(
        device
    )
    test_targets = add_noise_to_embeddings(test_embeddings, args.noise_level).to(device)

    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # Training history
    history = {"train_loss": [], "val_loss": [], "cosine_similarity": []}

    # Early stopping variables
    best_val_loss = float("inf")
    early_stopping_counter = 0
    best_model_path = os.path.join(args.save_dir, f"{args.model_name}_best.pth")

    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(
            model,
            train_dataloader,
            train_targets,
            optimizer,
            criterion,
            device,
            args.batch_size,
        )

        # Evaluate
        val_loss, val_metrics = evaluate(
            model, test_dataloader, test_targets, criterion, device, args.batch_size
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["cosine_similarity"].append(val_metrics["cosine_similarity"])

        # Print epoch results
        epoch_time = time.time() - epoch_start_time
        print(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Time: {epoch_time:.2f}s - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Cosine Sim: {val_metrics['cosine_similarity']:.4f}"
        )

        # Log to Weights & Biases
        if args.wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "cosine_similarity": val_metrics["cosine_similarity"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0

            # Save best model
            model.save(best_model_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        else:
            early_stopping_counter += 1
            print(
                f"EarlyStopping counter: {early_stopping_counter} out of {args.early_stopping}"
            )

            if early_stopping_counter >= args.early_stopping:
                print("Early stopping triggered")
                break

    # Training complete
    total_time = time.time() - start_time
    print(f"Training complete! Total time: {total_time:.2f}s")

    # Final model save
    final_model_path = os.path.join(args.save_dir, f"{args.model_name}_final.pth")
    model.save(final_model_path)
    print(f"Final model saved at {final_model_path}")

    # Save training history
    history_path = os.path.join(args.save_dir, f"{args.model_name}_history.pt")
    torch.save(history, history_path)
    print(f"Training history saved at {history_path}")

    # Plot training history if specified
    if args.plot_history:
        plot_path = os.path.join(
            args.save_dir, f"{args.model_name}_training_history.png"
        )
        plot_training_history(history, save_path=plot_path)

    # Close Weights & Biases
    if args.wandb:
        wandb.finish()

    return model, history


if __name__ == "__main__":
    args = parse_args()
    train(args)
