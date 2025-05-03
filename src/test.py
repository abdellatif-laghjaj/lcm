import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
import matplotlib.pyplot as plt
import spacy
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
import time
from typing import List, Tuple, Dict, Optional, Union

from baselcm import BaseLCM, DiffusionLCM, SonarEncoder
from utils import compute_metrics, load_config, cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Testing and evaluation for LCM")

    # Model parameters
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--config_path", type=str, help="Path to the model configuration file"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["base", "diffusion"],
        default="base",
        help="Type of LCM model: base or diffusion",
    )

    # Testing parameters
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument(
        "--noise_level", type=float, default=0.05, help="Noise level for testing"
    )
    parser.add_argument(
        "--sequence_length", type=int, default=5, help="Length of concept sequences"
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=10,
        help="Number of diffusion steps for DiffusionLCM",
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
        default=1000,
        help="Max rows to use from dataset for testing",
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

    # Inference parameters
    parser.add_argument("--input_text", type=str, help="Input text for inference")
    parser.add_argument(
        "--inference_mode",
        choices=["continuation", "denoising", "generation"],
        default="continuation",
        help="Mode for inference",
    )
    parser.add_argument(
        "--num_outputs", type=int, default=1, help="Number of outputs to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for generation"
    )

    # Other parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument("--visualize", action="store_true", help="Visualize results")

    return parser.parse_args()


def load_model_and_config(args):
    """Load model and configuration based on command line arguments.

    Args:
        args: Command line arguments

    Returns:
        tuple: (model, config)
    """
    # Load configuration
    if args.config_path and os.path.exists(args.config_path):
        config = load_config(args.config_path)
    else:
        # Try to find config based on model path
        possible_config = args.model_path.replace(".pth", "_config.json")
        if os.path.exists(possible_config):
            config = load_config(possible_config)
            print(f"Found and loaded config from {possible_config}")
        else:
            # Default configuration
            print("No configuration found, using default values")
            config = {
                "input_dim": 512,
                "hidden_dim": 768,
                "num_heads": 12,
                "num_layers": 6,
                "ff_dim": 3072,
                "dropout_rate": 0.1,
                "model_type": args.model_type,
                "diffusion_steps": args.diffusion_steps,
            }

    # Determine model type (base or diffusion)
    model_type = args.model_type
    if "model_type" in config:
        model_type = config["model_type"]

    # Get model dimensions
    input_dim = config.get("input_dim", 512)
    hidden_dim = config.get("hidden_dim", 768)
    num_heads = config.get("num_heads", 12)
    num_layers = config.get("num_layers", 6)
    ff_dim = config.get("ff_dim", 3072)
    dropout_rate = config.get("dropout_rate", 0.1)
    diffusion_steps = config.get("diffusion_steps", 10)

    # Initialize and load the appropriate model type
    print(f"Loading {model_type.upper()} LCM model...")
    if model_type == "diffusion":
        model = DiffusionLCM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            output_dim=input_dim,  # Output dim must match input for concept embeddings
            diffusion_steps=diffusion_steps,
            dropout_rate=dropout_rate,
        )
    else:
        model = BaseLCM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            output_dim=input_dim,  # Output dim must match input
            dropout_rate=dropout_rate,
        )

    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load scaling parameters if available
    if "prenet_mean" in checkpoint:
        model.prenet.scaler_mean = checkpoint.get("prenet_mean", 0.0)
        model.prenet.scaler_std = checkpoint.get("prenet_std", 1.0)
        model.postnet.scaler_mean = checkpoint.get("postnet_mean", 0.0)
        model.postnet.scaler_std = checkpoint.get("postnet_std", 1.0)

    model.to(args.device)
    model.eval()

    print(f"Model loaded from {args.model_path}")
    return model, config


class ConceptSequenceDataset(torch.utils.data.Dataset):
    """Dataset for concept sequences for testing.

    Creates sequences of concept embeddings for testing the LCM model.
    """

    def __init__(self, embeddings, seq_length):
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


def prepare_test_data(args, encoder):
    """Prepare test data for evaluation.

    Args:
        args: Command line arguments
        encoder: Initialized SonarEncoder

    Returns:
        tuple: (concept_embeddings, sentences)
    """
    device = args.device

    # Load dataset
    print(f"Loading test dataset: {args.dataset}")
    try:
        # Handle different datasets
        if args.dataset.lower() == "wikitext":
            dataset_name = "wikitext" if args.dataset_config else "wikitext-103-v1"
            raw_dataset = load_dataset(dataset_name, args.dataset_config, split="test")

        elif args.dataset.lower() == "bookcorpus":
            raw_dataset = load_dataset("bookcorpus", split="test")
            # Bookcorpus doesn't have a test split, so use a part of train
            raw_dataset = load_dataset("bookcorpus", split="train")

        elif args.dataset.lower() == "custom":
            # For custom datasets, load from the provided path
            if not args.dataset_config:
                raise ValueError("Must provide dataset_config path for custom dataset")
            raw_dataset = load_dataset(args.dataset_config, split="test")

        else:
            # Try to load the dataset as specified
            if args.trust_remote_code:
                raw_dataset = load_dataset(
                    args.dataset,
                    args.dataset_config,
                    split="test",
                    trust_remote_code=True,
                )
            else:
                raw_dataset = load_dataset(
                    args.dataset, args.dataset_config, split="test"
                )

        # Limit data for testing if specified
        if args.data_limit > 0:
            raw_dataset = raw_dataset.select(
                range(min(args.data_limit, len(raw_dataset)))
            )

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using WikiText-103 as fallback dataset...")
        raw_dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")

        if args.data_limit > 0:
            raw_dataset = raw_dataset.select(
                range(min(args.data_limit, len(raw_dataset)))
            )

    print(f"Loaded {len(raw_dataset)} examples from dataset")

    # Initialize or load spaCy for sentence splitting
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

    # Process in batches to avoid memory issues
    batch_size = 100
    for i in tqdm(range(0, len(raw_dataset), batch_size), desc="Processing Texts"):
        batch = raw_dataset[i : min(i + batch_size, len(raw_dataset))]
        for text in batch[args.text_column]:
            sentences = split_into_sentences(text)
            all_sentences.extend(sentences)

    print(f"Extracted {len(all_sentences)} sentences from dataset")

    # We need at least sequence_length + 1 sentences
    min_required = args.sequence_length + 1
    if len(all_sentences) < min_required:
        print(
            f"Warning: Not enough sentences ({len(all_sentences)}) for sequence length {args.sequence_length}"
        )
        print("Using raw texts as sentences")
        all_sentences = [
            text
            for text in raw_dataset[args.text_column]
            if isinstance(text, str) and text.strip()
        ]

    # Encode all sentences to concept embeddings
    print("Encoding sentences to concept embeddings...")
    encoder_batch_size = args.batch_size

    concept_embeddings = encoder.encode(
        all_sentences[
            : min(10000, len(all_sentences))
        ],  # Limit to 10k sentences for testing
        lang=args.lang,
        batch_size=encoder_batch_size,
    ).to(device)

    print(f"Generated {concept_embeddings.shape} concept embeddings for testing")

    return concept_embeddings, all_sentences


def evaluate_model(model, test_embeddings, args):
    """Evaluate the model on test data.

    Args:
        model: The LCM model to evaluate
        test_embeddings: The test concept embeddings
        args: Command line arguments

    Returns:
        dict: Evaluation results
    """
    device = args.device

    # Create dataset and dataloader for testing
    test_dataset = ConceptSequenceDataset(test_embeddings, args.sequence_length)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Loss function for evaluation
    criterion = nn.MSELoss()

    # Evaluation results
    results = {
        "mse_loss": 0.0,
        "cosine_similarity": 0.0,
        "predictions": [],
        "targets": [],
        "context": [],
    }

    # Evaluate model
    print("Evaluating model...")
    model.eval()
    all_preds = []
    all_targets = []
    all_contexts = []

    with torch.no_grad():
        for x_seq, y in tqdm(test_dataloader, desc="Testing"):
            # Move data to device
            x_seq = x_seq.to(device)
            y = y.to(device)

            # Forward pass
            if isinstance(model, DiffusionLCM) and args.diffusion_steps > 0:
                # For diffusion model, use the generation capability
                y_pred = model.generate(x_seq, steps=args.diffusion_steps)
            else:
                # Standard forward pass for base model
                y_pred = model(x_seq)

            # Store results
            all_preds.append(y_pred.cpu())
            all_targets.append(y.cpu())
            all_contexts.append(x_seq.cpu())

    # Concatenate results
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate metrics
    metrics = compute_metrics(all_preds, all_targets)
    mse_loss = nn.MSELoss()(all_preds, all_targets).item()

    # Store results
    results["mse_loss"] = mse_loss
    results["cosine_similarity"] = metrics["cosine_similarity"]

    # Store a sample of predictions for visualization
    sample_size = min(10, len(all_preds))
    results["predictions"] = all_preds[:sample_size].tolist()
    results["targets"] = all_targets[:sample_size].tolist()

    # Print results
    print(f"Evaluation Results:")
    print(f"MSE Loss: {mse_loss:.6f}")
    print(f"Cosine Similarity: {metrics['cosine_similarity']:.4f}")

    return results


def run_inference(model, encoder, args):
    """Run inference with the model.

    Args:
        model: The LCM model for inference
        encoder: SONAR encoder for text
        args: Command line arguments

    Returns:
        dict: Inference results
    """
    device = args.device

    if not args.input_text:
        print("No input text provided for inference")
        return {}

    print(f"Running inference in {args.inference_mode} mode")
    print(f"Input text: {args.input_text}")

    # Process input text
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        import subprocess

        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    # Split input text into sentences
    doc = nlp(args.input_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if not sentences:
        print("No valid sentences found in input text")
        return {}

    print(f"Found {len(sentences)} sentences in input text")

    # Encode sentences
    input_embeddings = encoder.encode(sentences, lang=args.lang).to(device)

    # Prepare sequence for model input
    if len(sentences) < args.sequence_length:
        print(
            f"Warning: Input has fewer sentences ({len(sentences)}) than sequence length ({args.sequence_length})"
        )
        seq_len = len(sentences)
    else:
        seq_len = args.sequence_length

    # Get the last seq_len sentences as context
    context_embeddings = input_embeddings[-seq_len:].unsqueeze(0)  # Add batch dimension

    # Run inference based on mode
    results = {
        "input_text": args.input_text,
        "sentences": sentences,
        "mode": args.inference_mode,
        "outputs": [],
    }

    # Inference modes
    if args.inference_mode == "continuation":
        print("Generating continuation...")

        for i in range(args.num_outputs):
            if isinstance(model, DiffusionLCM):
                # Use diffusion model's generation capability
                next_embedding = model.generate(context_embeddings)
            else:
                # Standard prediction with base model
                next_embedding = model(context_embeddings)

            # Store result
            results["outputs"].append(
                {"embedding": next_embedding.cpu().tolist(), "mode": "continuation"}
            )

    elif args.inference_mode == "denoising":
        print("Denoising concept...")

        # Take the last embedding and add noise
        target_embedding = input_embeddings[-1].unsqueeze(0)  # Add batch dimension
        noisy_embedding = (
            target_embedding + torch.randn_like(target_embedding) * args.noise_level
        )

        # Context is all sentences except the last one
        if len(sentences) > 1:
            context_for_denoising = input_embeddings[:-1][
                -args.sequence_length :
            ].unsqueeze(0)
        else:
            context_for_denoising = torch.zeros(
                (1, 1, input_embeddings.size(-1)), device=device
            )

        # Denoise
        if isinstance(model, DiffusionLCM):
            # Use diffusion model's denoising capability
            denoised_embedding = model.denoise_step(
                noisy_embedding[0], context_for_denoising, 0
            )
        else:
            # For base model, just process through the model
            full_input = torch.cat(
                [context_for_denoising, noisy_embedding.unsqueeze(1)], dim=1
            )
            output = model(full_input)
            denoised_embedding = output

        # Compute similarity to original
        similarity = cosine_similarity(
            denoised_embedding.cpu().numpy(), target_embedding.cpu().numpy()
        )[0][0]

        # Print result
        print(f"Denoising Results:")
        print(f"Cosine Similarity to Original: {similarity:.4f}")

        # Store result
        results["outputs"].append(
            {
                "embedding": denoised_embedding.cpu().tolist(),
                "original": target_embedding.cpu().tolist(),
                "similarity": similarity,
                "mode": "denoising",
            }
        )

    elif args.inference_mode == "generation":
        print("Generation from context...")

        if not isinstance(model, DiffusionLCM):
            print("Warning: Generation mode works best with DiffusionLCM")

        # Generate multiple outputs with different random seeds
        for i in range(args.num_outputs):
            # Set random seed for reproducibility but different outputs
            torch.manual_seed(42 + i)

            if isinstance(model, DiffusionLCM):
                # Use diffusion model with more steps for generation
                generated_embedding = model.generate(
                    context_embeddings, steps=args.diffusion_steps
                )
            else:
                # Add some randomness to output for base model
                output = model(context_embeddings)
                noise = torch.randn_like(output) * (args.temperature * 0.1)
                generated_embedding = output + noise

            # Store result
            results["outputs"].append(
                {
                    "embedding": generated_embedding.cpu().tolist(),
                    "seed": 42 + i,
                    "mode": "generation",
                }
            )

    return results


def visualize_results(results, output_dir):
    """Visualize evaluation or inference results.

    Args:
        results: Results dictionary
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check what type of results we have
    if "mse_loss" in results:
        # Evaluation results
        try:
            from sklearn.decomposition import PCA

            # Convert predictions and targets to numpy
            predictions = np.array(results["predictions"])
            targets = np.array(results["targets"])

            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=2)
            combined = np.vstack([predictions, targets])
            embedded = pca.fit_transform(combined)

            # Split back to predictions and targets
            pred_embedded = embedded[: len(predictions)]
            target_embedded = embedded[len(predictions) :]

            # Plot
            plt.figure(figsize=(10, 8))
            plt.scatter(
                pred_embedded[:, 0], pred_embedded[:, 1], c="red", label="Predictions"
            )
            plt.scatter(
                target_embedded[:, 0], target_embedded[:, 1], c="blue", label="Targets"
            )

            # Draw lines connecting corresponding points
            for i in range(len(pred_embedded)):
                plt.plot(
                    [pred_embedded[i, 0], target_embedded[i, 0]],
                    [pred_embedded[i, 1], target_embedded[i, 1]],
                    "k-",
                    alpha=0.3,
                )

            plt.title(
                f"Predictions vs Targets (Cosine Sim: {results['cosine_similarity']:.4f})"
            )
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.legend()

            # Save
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "pred_vs_target.png"))
            print(
                f"Visualization saved to {os.path.join(output_dir, 'pred_vs_target.png')}"
            )

        except Exception as e:
            print(f"Error generating visualization: {e}")

    # Save results as JSON
    try:
        # Convert numpy arrays to lists for JSON serialization
        json_compatible_results = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                json_compatible_results[k] = v.tolist()
            else:
                json_compatible_results[k] = v

        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(json_compatible_results, f, indent=2)

        print(f"Results saved to {os.path.join(output_dir, 'results.json')}")
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    """Main function for testing and inference."""
    args = parse_args()

    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load model and configuration
    model, config = load_model_and_config(args)

    # Initialize encoder
    print("Initializing SONAR encoder...")
    encoder = SonarEncoder(device=args.device)

    # Results to store
    results = {}

    # Run inference if input text is provided
    if args.input_text:
        inference_results = run_inference(model, encoder, args)
        results.update(inference_results)

    # Test on dataset if specified
    if args.dataset:
        # Prepare test data
        test_embeddings, test_sentences = prepare_test_data(args, encoder)

        # Evaluate model
        eval_results = evaluate_model(model, test_embeddings, args)
        results.update(eval_results)

    # Visualize results if specified
    if args.visualize and results:
        visualize_results(results, args.output_dir)

    # If neither test data nor input text is provided
    if not args.dataset and not args.input_text:
        print(
            "No dataset or input text provided. Please specify --dataset or --input_text"
        )
        print("Example usage:")
        print(
            "  Testing: python test.py --model_path saved_models/lcm_model_best.pth --dataset wikitext --data_limit 1000"
        )
        print(
            "  Inference: python test.py --model_path saved_models/lcm_model_best.pth --input_text 'This is a test sentence.' --inference_mode continuation"
        )

    print("Testing complete!")
    return results


if __name__ == "__main__":
    start_time = time.time()
    results = main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
