import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import wandb
from tqdm import tqdm
import numpy as np
import argparse
from model import LCM, DiffusionLCM, TwoTowerDiffusionLCM


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    num_epochs=5,
    model_type="base",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in train_bar:
            # Move batch to device
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)

            # For diffusion models, loss is already calculated in the forward pass
            if model_type == "base":
                loss = criterion(outputs[0], labels)
            else:
                loss = outputs[1]  # Loss is second output for diffusion models

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix({"loss": loss.item()})

            # Log to wandb
            wandb.log({"train_loss": loss.item(), "epoch": epoch})

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)

                if model_type == "base":
                    outputs = model(**inputs, labels=labels)
                    loss = criterion(outputs[0], labels)
                else:
                    outputs = model(**inputs, labels=labels)
                    loss = outputs[1]  # Loss is second output for diffusion models

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch})

        print(f"\nEpoch {epoch+1}:")
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average validation loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{model_type}_model.pt")
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a Large Concept Model")
    parser.add_argument(
        "--model_type",
        type=str,
        default="base",
        choices=["base", "diffusion", "two_tower"],
        help="Type of LCM model to train",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=10,
        help="Number of diffusion steps (only for diffusion models)",
    )
    parser.add_argument(
        "--encoder_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Pretrained encoder model to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cnn_dailymail",
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        default="3.0.0",
        help="Version of the dataset",
    )
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(
        project="lcm-practical",
        name=f"{args.model_type}-lcm-training",
        config=vars(args),
    )

    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, args.dataset_version)

    # Initialize model based on type
    print(f"Initializing {args.model_type.upper()}-LCM model...")
    if args.model_type == "base":
        model = LCM(encoder_model=args.encoder_model)
    elif args.model_type == "diffusion":
        model = DiffusionLCM(
            encoder_model=args.encoder_model, diffusion_steps=args.diffusion_steps
        )
    elif args.model_type == "two_tower":
        model = TwoTowerDiffusionLCM(
            encoder_model=args.encoder_model, diffusion_steps=args.diffusion_steps
        )

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)

    # Prepare data loaders
    def collate_fn(batch):
        # Tokenize and prepare batch
        inputs = tokenizer(
            [item["article"] for item in batch],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenizer(
            [item["highlights"] for item in batch],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]
        inputs["labels"] = labels
        return inputs

    train_loader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Setup training
    criterion = (
        nn.MSELoss() if args.model_type == "base" else None
    )  # Diffusion models handle loss internally
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Train
    print("Starting training...")
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs=args.epochs,
        model_type=args.model_type,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
