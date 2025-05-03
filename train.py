import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import numpy as np
import argparse
import os
import time
from model import LCM, DiffusionLCM, TwoTowerDiffusionLCM


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler=None,
    num_epochs=5,
    model_type="base",
    gradient_accumulation_steps=1,
    checkpoint_dir="checkpoints",
    checkpoint_interval=30,  # minutes
    max_val_samples=None,
    fp16=False,
    clip_grad_norm=1.0,
    use_wandb=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_val_loss = float("inf")

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Check if mixed precision is available and enabled
    use_fp16 = fp16 and device.type == "cuda" and torch.cuda.is_available()
    if fp16 and not use_fp16:
        print(
            "Warning: Mixed precision (fp16) requested but CUDA is not available. Using standard precision instead."
        )

    # Initialize scaler for mixed precision training only if available
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    # Initialize checkpoint timing
    last_checkpoint_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch

        # Training loop
        for i, batch in enumerate(train_bar):
            # Move batch to device
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            # Forward pass with mixed precision if enabled
            if use_fp16:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(**inputs, labels=labels)
                    # For diffusion models, loss is already calculated in the forward pass
                    if model_type == "base":
                        loss = criterion(outputs[0], labels)
                    else:
                        loss = outputs[1]  # Loss is second output for diffusion models

                    # Apply gradient accumulation
                    loss = loss / gradient_accumulation_steps

                # Backward pass with scaling
                scaler.scale(loss).backward()

                # Step optimizer and scaler if gradient accumulation steps reached
                if (i + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    if clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), clip_grad_norm
                        )

                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
            else:
                # Standard precision training
                outputs = model(**inputs, labels=labels)
                if model_type == "base":
                    loss = criterion(outputs[0], labels)
                else:
                    loss = outputs[1]

                # Apply gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()

                # Step optimizer if gradient accumulation steps reached
                if (i + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    if clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), clip_grad_norm
                        )

                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

            # For logging, we use the full loss value
            full_loss = loss.item() * gradient_accumulation_steps
            total_loss += full_loss
            train_bar.set_postfix({"loss": full_loss})

            # Log to wandb periodically to reduce overhead
            if use_wandb and i % 10 == 0:
                wandb.log(
                    {
                        "train_loss": full_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

            # Save checkpoint based on time interval
            current_time = time.time()
            if (current_time - last_checkpoint_time) / 60 >= checkpoint_interval:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"{model_type}_epoch{epoch+1}_step{i+1}.pt"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "step": i,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": full_loss,
                        "scheduler": scheduler.state_dict() if scheduler else None,
                    },
                    checkpoint_path,
                )
                print(f"\nIntermediate checkpoint saved to {checkpoint_path}")
                last_checkpoint_time = current_time

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            # Use a subset of validation data if specified
            val_dataset = val_loader.dataset
            if max_val_samples and max_val_samples < len(val_dataset):
                indices = np.random.choice(
                    len(val_dataset), max_val_samples, replace=False
                )
                val_dataset = Subset(val_dataset, indices)
                val_bar = tqdm(
                    DataLoader(
                        val_dataset,
                        batch_size=val_loader.batch_size,
                        collate_fn=val_loader.collate_fn,
                    ),
                    desc="Validation",
                )
            else:
                val_bar = tqdm(val_loader, desc="Validation")

            for batch in val_bar:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)

                outputs = model(**inputs, labels=labels)
                if model_type == "base":
                    loss = criterion(outputs[0], labels)
                else:
                    loss = outputs[1]  # Loss is second output for diffusion models

                val_loss += loss.item()
                val_bar.set_postfix({"val_loss": loss.item()})

        val_size = len(val_dataset) if max_val_samples else len(val_loader)
        avg_val_loss = val_loss / val_size
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss_epoch": avg_train_loss,
                    "val_loss_epoch": avg_val_loss,
                }
            )

        print(f"\nEpoch {epoch+1}:")
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average validation loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_dir, f"{model_type}_best_model.pt"),
            )
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")

        # Always save the latest model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_val_loss,
                "scheduler": scheduler.state_dict() if scheduler else None,
            },
            os.path.join(checkpoint_dir, f"{model_type}_latest.pt"),
        )


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a Large Concept Model")
    # Model parameters
    parser.add_argument(
        "--model_type",
        type=str,
        default="base",
        choices=["base", "diffusion", "two_tower"],
        help="Type of LCM model to train",
    )
    parser.add_argument(
        "--encoder_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Pretrained encoder model to use",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=10,
        help="Number of diffusion steps (only for diffusion models)",
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=500, help="Learning rate warmup steps"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Use mixed precision training"
    )
    parser.add_argument(
        "--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm"
    )

    # Dataset parameters
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
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Max number of training samples (for debug/quick runs)",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="Max number of validation samples",
    )

    # Checkpoint parameters
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=30,
        help="Save checkpoint every X minutes",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from this checkpoint file",
    )

    # Performance options
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of worker threads for data loading",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory for faster data transfer (only useful with CUDA)",
    )

    # Wandb configuration
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Whether to use Weights & Biases for logging",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode (online, offline, disabled)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="lcm-practical",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity name (username or team name)",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )

    args = parser.parse_args()

    # Initialize wandb if enabled
    use_wandb = args.use_wandb
    if use_wandb:
        run_name = args.wandb_name or f"{args.model_type}-lcm-training"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            mode=args.wandb_mode,
        )
    else:
        print("Weights & Biases logging is disabled")

    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, args.dataset_version)

    # Optionally limit the number of training samples (for debugging)
    if args.max_train_samples is not None and args.max_train_samples < len(
        dataset["train"]
    ):
        dataset["train"] = dataset["train"].select(range(args.max_train_samples))
        print(f"Limited training dataset to {args.max_train_samples} samples")

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
            max_length=512,  # Add max length to prevent excessive memory usage
            return_tensors="pt",
        )
        labels = tokenizer(
            [item["highlights"] for item in batch],
            padding=True,
            truncation=True,
            max_length=128,  # Add max length for targets
            return_tensors="pt",
        )["input_ids"]
        inputs["labels"] = labels
        return inputs

    # Check if CUDA is available for pin_memory and num_workers
    use_cuda = torch.cuda.is_available()
    pin_memory = args.pin_memory and use_cuda

    # Reduce num_workers if not on a machine that can handle it
    num_workers = args.num_workers if use_cuda else 0

    train_loader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Setup training
    criterion = (
        nn.MSELoss() if args.model_type == "base" else None
    )  # Diffusion models handle loss internally
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Add learning rate scheduler
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler" in checkpoint and checkpoint["scheduler"] and scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            print(f"Starting from epoch {start_epoch + 1}")

    # Train
    print("Starting training...")
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler=scheduler,
        num_epochs=args.epochs,
        model_type=args.model_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        max_val_samples=args.max_val_samples,
        fp16=args.fp16,
        clip_grad_norm=args.clip_grad_norm,
        use_wandb=use_wandb,
    )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
