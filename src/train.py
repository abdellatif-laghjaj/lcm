import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import argparse
import wandb
import os
import time
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt
import spacy
from typing import List, Tuple, Dict, Optional, Union

from baselcm import BaseLCM, DiffusionLCM, SonarEncoder
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
        "--input_dim", type=int, default=512, help="Input dimension for the model"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=768, help="Hidden dimension for the model"
    )
    parser.add_argument(
        "--num_heads", type=int, default=12, help="Number of heads for the model"
    )
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--ff_dim", type=int, default=3072, help="Feedforward dimension for the model"
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.1,
        help="Dropout rate for regularization",
    )

    # Model type selection
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["base", "diffusion"],
        default="base",
        help="Type of LCM model: base or diffusion",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=10,
        help="Number of diffusion steps (only for diffusion model)",
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--sequence_length", type=int, default=5, help="Length of concept sequences"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate for training"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.05,
        help="Noise level for diffusion training",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Fraction of data to use for testing",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--early_stopping", type=int, default=5, help="Early stopping patience"
    )

    # Data parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Dataset name: wikitext, bookcorpus, custom",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-103-v1",
        help="Configuration for the dataset",
    )
    parser.add_argument(
        "--data_limit",
        type=int,
        default=-1,
        help="Max rows to use from dataset (-1 for all data)",
    )
    parser.add_argument(
        "--text_column", type=str, default="text", help="Text column in the dataset"
    )
    parser.add_argument(
        "--lang", type=str, default="en", help="Language for the dataset"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for dataset loading",
    )
    parser.add_argument(
        "--min_sent_length", type=int, default=8, help="Minimum sentence length to keep"
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
        "--model_name", type=str, default="lcm_model", help="Name of the model"
    )
    parser.add_argument(
        "--plot_history", action="store_true", help="Plot training history"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Save model every N epochs (0 to disable)",
    )

    return parser.parse_args()


# Classes and functions to process sentences into concept sequence batches
class ConceptSequenceDataset(Dataset):
    """Dataset for handling concept sequences for LCM training.

    Creates sequences of concept embeddings where each sequence is a list of
    consecutive sentences to predict the next sentence in the sequence.
    """

    def __init__(self, embeddings, seq_length):
        """Initialize the dataset.

        Args:
            embeddings: Tensor of sentence concept embeddings
            seq_length: Length of sequences to create
        """
        self.embeddings = embeddings
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.embeddings) - self.seq_length)

    def __getitem__(self, idx):
        # Get sequence of concepts (sentences)
        x_seq = self.embeddings[idx : idx + self.seq_length]

        # Get target (next concept/sentence)
        y = self.embeddings[idx + self.seq_length]

        return x_seq, y


# Function to move data to device
def to_device(data, device):
    """Move data to the specified device.

    Args:
        data: Data to move (can be tensor, list, tuple)
        device: Target device

    Returns:
        Data on the specified device
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


def train_epoch(model, dataloader, optimizer, criterion, device, accumulation_steps=1):
    """Train model for one epoch.

    Args:
        model: The LCM model to train
        dataloader: DataLoader for training data
        optimizer: Model optimizer
        criterion: Loss function
        device: Device to use
        accumulation_steps: Number of gradient accumulation steps

    Returns:
        float: Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    total_steps = len(dataloader)

    pbar = tqdm(enumerate(dataloader), total=total_steps, desc="Training")

    for step, (x_seq, y) in pbar:
        # Move data to device
        x_seq = to_device(x_seq, device)
        y = to_device(y, device)

        # Forward pass
        y_pred = model(x_seq)
        loss = criterion(y_pred, y)

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights only after accumulation steps
        if (step + 1) % accumulation_steps == 0 or (step + 1) == total_steps:
            optimizer.step()
            optimizer.zero_grad()

        # Update running loss
        running_loss += loss.item() * accumulation_steps
        pbar.set_postfix({"loss": loss.item() * accumulation_steps})

    return running_loss / total_steps


def train_diffusion_epoch(
    model, dataloader, optimizer, criterion, device, noise_level, accumulation_steps=1
):
    """Train diffusion LCM for one epoch.

    Args:
        model: The DiffusionLCM model to train
        dataloader: DataLoader for training data
        optimizer: Model optimizer
        criterion: Loss function
        device: Device to use
        noise_level: Maximum noise level to add
        accumulation_steps: Number of gradient accumulation steps

    Returns:
        float: Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    total_steps = len(dataloader)

    pbar = tqdm(enumerate(dataloader), total=total_steps, desc="Training")

    for step, (x_seq, y) in pbar:
        # Move data to device
        x_seq = to_device(x_seq, device)
        y = to_device(y, device)

        # Add random noise to target
        noise_scale = torch.rand(y.size(0), 1, device=device) * noise_level
        noisy_y = y + torch.randn_like(y) * noise_scale

        # Random timestep for each example
        batch_size = y.size(0)
        timesteps = torch.randint(
            0, model.diffusion_steps, (batch_size,), device=device
        )

        # Forward pass with denoising
        optimizer.zero_grad()

        # Denoise step
        y_pred = torch.zeros_like(y)
        for i in range(batch_size):
            t = timesteps[i].item()
            y_pred[i] = model.denoise_step(noisy_y[i], x_seq[i : i + 1], t)

        # Compute loss against clean target
        loss = criterion(y_pred, y)

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights only after accumulation steps
        if (step + 1) % accumulation_steps == 0 or (step + 1) == total_steps:
            optimizer.step()
            optimizer.zero_grad()

        # Update running loss
        running_loss += loss.item() * accumulation_steps
        pbar.set_postfix({"loss": loss.item() * accumulation_steps})

    return running_loss / total_steps


def evaluate(model, dataloader, criterion, device):
    """Evaluate LCM model.

    Args:
        model: LCM model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to use

    Returns:
        Tuple: (average loss, metrics dictionary)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_seq, y in dataloader:
            # Move data to device
            x_seq = to_device(x_seq, device)
            y = to_device(y, device)

            # Forward pass
            y_pred = model(x_seq)
            loss = criterion(y_pred, y)

            # Update running loss
            running_loss += loss.item()

            # Store predictions and targets for metrics
            all_preds.append(y_pred.cpu())
            all_targets.append(y.cpu())

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    metrics = compute_metrics(all_preds, all_targets)

    return running_loss / len(dataloader), metrics


def train(args):
    """Main training function for LCM.

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

    # Initialize model based on selected type
    print(f"Initializing {args.model_type.upper()} LCM model...")
    if args.model_type == "diffusion":
        model = DiffusionLCM(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            output_dim=args.input_dim,  # Output dim must match input for concept embeddings
            diffusion_steps=args.diffusion_steps,
            dropout_rate=args.dropout_rate,
        ).to(device)
    else:
        model = BaseLCM(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            output_dim=args.input_dim,  # Output dim must match input for concept embeddings
            dropout_rate=args.dropout_rate,
        ).to(device)

    # Count model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")

    # Initialize SONAR encoder
    print("Initializing SONAR encoder...")
    encoder = SonarEncoder(device=device)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    try:
        # Handle different datasets
        if args.dataset.lower() == "wikitext":
            dataset_name = "wikitext" if args.dataset_config else "wikitext-103-v1"
            raw_dataset = load_dataset(dataset_name, args.dataset_config, split="train")

        elif args.dataset.lower() == "bookcorpus":
            raw_dataset = load_dataset("bookcorpus", split="train")

        elif args.dataset.lower() == "custom":
            # For custom datasets, load from the provided path
            if not args.dataset_config:
                raise ValueError("Must provide dataset_config path for custom dataset")
            raw_dataset = load_dataset(args.dataset_config, split="train")

        else:
            # Try to load the dataset as specified
            if args.trust_remote_code:
                raw_dataset = load_dataset(
                    args.dataset,
                    args.dataset_config,
                    split="train",
                    trust_remote_code=True,
                )
            else:
                raw_dataset = load_dataset(
                    args.dataset, args.dataset_config, split="train"
                )

        # Use all data unless explicitly limited
        if args.data_limit > 0:
            raw_dataset = raw_dataset.select(
                range(min(args.data_limit, len(raw_dataset)))
            )
            print(f"Using {len(raw_dataset)} rows from dataset")
        else:
            print(f"Using all {len(raw_dataset)} rows from dataset")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using WikiText-103 as fallback dataset...")
        raw_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

        if args.data_limit > 0:
            raw_dataset = raw_dataset.select(
                range(min(args.data_limit, len(raw_dataset)))
            )

    # Initialize or load spaCy for sentence splitting
    print("Initializing NLP model for sentence splitting...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        import subprocess

        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    # Function to split text into sentences
    def split_into_sentences(text):
        if not text or not isinstance(text, str):
            return []
        try:
            doc = nlp(text)
            # Only keep sentences with minimum length to filter out headers, code, etc.
            sentences = [
                sent.text.strip()
                for sent in doc.sents
                if sent.text.strip() and len(sent.text.split()) >= args.min_sent_length
            ]
            return sentences
        except Exception as e:
            print(f"Error processing text: {e}")
            return []

    # Process the dataset to extract sentences
    print("Processing dataset into sentences...")
    all_sentences = []

    # Process in batches to avoid memory issues with very large datasets
    batch_size = 1000
    for i in tqdm(range(0, len(raw_dataset), batch_size), desc="Processing Texts"):
        batch = raw_dataset[i : min(i + batch_size, len(raw_dataset))]
        for text in tqdm(
            batch[args.text_column], desc="Extracting Sentences", leave=False
        ):
            sentences = split_into_sentences(text)
            all_sentences.extend(sentences)

    print(f"Extracted {len(all_sentences)} sentences from dataset")

    # Encode all sentences to concept embeddings
    print("Encoding sentences to concept embeddings...")
    # Process in manageable batches
    encoder_batch_size = min(32, args.batch_size)

    # Use custom chunking for very large datasets
    chunk_size = 100000  # Process this many sentences at a time
    all_embeddings = []

    for chunk_start in range(0, len(all_sentences), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(all_sentences))
        print(f"Processing sentences {chunk_start} to {chunk_end}...")

        chunk_sentences = all_sentences[chunk_start:chunk_end]
        chunk_embeddings = encoder.encode(
            chunk_sentences, lang=args.lang, batch_size=encoder_batch_size
        )
        all_embeddings.append(chunk_embeddings)

    # Concatenate all embeddings
    concept_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Generated {concept_embeddings.shape} concept embeddings")

    # Free up memory
    del encoder, all_sentences, all_embeddings
    torch.cuda.empty_cache()

    # Compute embedding statistics for normalization
    mean = torch.mean(concept_embeddings, dim=0)
    std = torch.std(concept_embeddings, dim=0)
    print(
        f"Embedding stats - Mean: {mean.mean().item():.4f}, Std: {std.mean().item():.4f}"
    )

    # Update model with statistics for normalization
    model.prenet.scaler_mean = mean.mean().item()
    model.prenet.scaler_std = std.mean().item()
    model.postnet.scaler_mean = mean.mean().item()
    model.postnet.scaler_std = std.mean().item()

    # Split data into train and validation sets
    train_embeddings, val_embeddings = split_train_test(
        concept_embeddings, test_size=args.test_size
    )

    # Create datasets for sequence prediction
    print("Creating sequence datasets...")
    train_dataset = ConceptSequenceDataset(train_embeddings, args.sequence_length)
    val_dataset = ConceptSequenceDataset(val_embeddings, args.sequence_length)

    print(
        f"Training sequences: {len(train_dataset)}, Validation sequences: {len(val_dataset)}"
    )

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,  # Drop incomplete batches
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False
    )

    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # MSE loss for concept embedding prediction
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

        # Train epoch
        if args.model_type == "diffusion":
            train_loss = train_diffusion_epoch(
                model,
                train_dataloader,
                optimizer,
                criterion,
                device,
                args.noise_level,
                args.gradient_accumulation_steps,
            )
        else:
            train_loss = train_epoch(
                model,
                train_dataloader,
                optimizer,
                criterion,
                device,
                args.gradient_accumulation_steps,
            )

        # Evaluate
        val_loss, val_metrics = evaluate(model, val_dataloader, criterion, device)

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
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
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

        # Save model periodically if requested
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.save_dir, f"{args.model_name}_epoch{epoch+1}.pth"
            )
            model.save(checkpoint_path)

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0

            # Save best model
            model.save(best_model_path)
            print(f"Saved best model with validation loss: {best_val_loss:.6f}")
        else:
            early_stopping_counter += 1
            print(
                f"Early stopping counter: {early_stopping_counter} out of {args.early_stopping}"
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
