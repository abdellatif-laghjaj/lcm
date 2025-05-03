import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
import matplotlib.pyplot as plt
import spacy
from tqdm.auto import tqdm
from datasets import load_dataset

from baselcm import BaseLCM, SonarEncoder
from utils import compute_metrics, load_config, cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Testing for BaseLCM")

    # Model parameters
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--config_path", type=str, help="Path to the model configuration file"
    )

    # Testing parameters
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for testing"
    )
    parser.add_argument(
        "--noise_level", type=float, default=0.05, help="Noise level for testing"
    )

    # Data parameters
    parser.add_argument(
        "--test_data", type=str, help="Path to test data or dataset name"
    )
    parser.add_argument(
        "--test_data_config", type=str, help="Configuration for the test dataset"
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
        default=100,
        help="Number of samples to use for testing",
    )
    parser.add_argument(
        "--use_sentences", action="store_true", help="Split text into sentences"
    )

    # Inference parameters
    parser.add_argument("--input_text", type=str, help="Input text for inference")
    parser.add_argument(
        "--inference_mode",
        choices=["denoising", "generation"],
        default="denoising",
        help="Mode for inference: denoising or generation",
    )
    parser.add_argument(
        "--noise_steps",
        type=int,
        default=10,
        help="Number of noise steps for generation",
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
    """Load model and configuration.

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
                "input_dim": 256,
                "hidden_dim": 512,
                "num_heads": 8,
                "num_layers": 6,
                "ff_dim": 2048,
                "output_dim": 256,
            }

    # Load model
    model = BaseLCM.load(
        path=args.model_path,
        input_dim=config.get("input_dim", 256),
        hidden_dim=config.get("hidden_dim", 512),
        num_heads=config.get("num_heads", 8),
        num_layers=config.get("num_layers", 6),
        ff_dim=config.get("ff_dim", 2048),
        output_dim=config.get("output_dim", 256),
        device=args.device,
    )

    print(f"Model loaded from {args.model_path}")
    return model, config


def test_on_dataset(model, encoder, args):
    """Test model on a dataset.

    Args:
        model: Trained model
        encoder: Text encoder
        args: Command line arguments

    Returns:
        dict: Test results
    """
    device = args.device
    model.eval()

    # Load dataset
    try:
        if args.test_data_config:
            dataset = load_dataset(args.test_data, args.test_data_config, split="test")
        else:
            dataset = load_dataset(args.test_data, split="test")

        dataset = dataset.select(range(min(args.data_sample, len(dataset))))
        print(f"Loaded {len(dataset)} samples from {args.test_data}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using a fallback dataset...")
        dataset = load_dataset(
            "oscar", "unshuffled_deduplicated_en", split="train"
        ).select(range(args.data_sample))

    # Process texts
    texts = dataset[args.text_column]

    # Split texts into sentences if specified
    if args.use_sentences:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")

        processed_texts = []
        for text in tqdm(texts, desc="Processing Texts"):
            doc = nlp(text)
            sentences = [sent.text for sent in doc.sents]
            processed_texts.extend(sentences)

        texts = processed_texts

    print(f"Processing {len(texts)} texts...")

    # Encode texts
    input_embeddings = encoder.encode(
        texts, lang=args.lang, batch_size=args.batch_size
    ).to(device)

    # Create noisy versions for testing denoising
    noisy_embeddings = (
        input_embeddings + torch.randn_like(input_embeddings) * args.noise_level
    )

    # Test denoising
    with torch.no_grad():
        denoised_embeddings = model(noisy_embeddings)

    # Compute metrics
    metrics = compute_metrics(denoised_embeddings, input_embeddings)

    print(f"Test Results:")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"Cosine Similarity: {metrics['cosine_similarity']:.4f}")

    # Save results
    results = {"metrics": metrics, "num_samples": len(texts)}

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "test_results.json")
        with open(results_path, "w") as f:
            json.dump(
                {
                    k: v if not isinstance(v, torch.Tensor) else v.tolist()
                    for k, v in results.items()
                },
                f,
                indent=4,
            )
        print(f"Test results saved to {results_path}")

    # Visualization
    if args.visualize:
        # Visualize a sample of the denoising results using PCA/t-SNE
        try:
            from sklearn.decomposition import PCA
            import numpy as np

            # Use a subset for visualization
            subset_size = min(100, len(input_embeddings))
            pca = PCA(n_components=2)

            # Apply PCA to original, noisy, and denoised embeddings
            embeddings_combined = (
                torch.cat(
                    [
                        input_embeddings[:subset_size],
                        noisy_embeddings[:subset_size],
                        denoised_embeddings[:subset_size],
                    ],
                    dim=0,
                )
                .cpu()
                .numpy()
            )

            embeddings_2d = pca.fit_transform(embeddings_combined)

            # Split back into original, noisy, and denoised
            original_2d = embeddings_2d[:subset_size]
            noisy_2d = embeddings_2d[subset_size : 2 * subset_size]
            denoised_2d = embeddings_2d[2 * subset_size :]

            # Plot
            plt.figure(figsize=(10, 8))
            plt.scatter(
                original_2d[:, 0],
                original_2d[:, 1],
                c="blue",
                label="Original",
                alpha=0.7,
            )
            plt.scatter(
                noisy_2d[:, 0], noisy_2d[:, 1], c="red", label="Noisy", alpha=0.3
            )
            plt.scatter(
                denoised_2d[:, 0],
                denoised_2d[:, 1],
                c="green",
                label="Denoised",
                alpha=0.5,
            )
            plt.title("PCA Visualization of Embeddings")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.legend()

            viz_path = os.path.join(args.output_dir, "embedding_visualization.png")
            plt.savefig(viz_path)
            print(f"Visualization saved to {viz_path}")
            plt.show()

        except Exception as e:
            print(f"Error in visualization: {e}")

    return results


def run_inference(model, encoder, args):
    """Run inference with the model.

    Args:
        model: Trained model
        encoder: Text encoder
        args: Command line arguments
    """
    device = args.device
    model.eval()

    if not args.input_text:
        print("No input text provided for inference")
        return

    print(f"Running inference in {args.inference_mode} mode")
    print(f"Input text: {args.input_text}")

    # Encode input text
    input_embedding = encoder.encode([args.input_text], lang=args.lang).to(device)

    if args.inference_mode == "denoising":
        # Add noise to input
        noisy_embedding = (
            input_embedding + torch.randn_like(input_embedding) * args.noise_level
        )

        # Denoise
        with torch.no_grad():
            denoised_embedding = model(noisy_embedding)

        # Compute metrics
        metrics = compute_metrics(denoised_embedding, input_embedding)

        print(f"Denoising Results:")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"Cosine Similarity: {metrics['cosine_similarity']:.4f}")

    elif args.inference_mode == "generation":
        # Implement progressive generation with noise sampling
        # Start with random noise and progressively denoise
        embedding_shape = input_embedding.shape

        # Start with gaussian noise
        current_embedding = torch.randn(embedding_shape, device=device)

        print("Generation process:")
        for step in tqdm(range(args.noise_steps), desc="Generation Steps"):
            # Apply model to current state
            with torch.no_grad():
                denoised = model(current_embedding)

            # Update current state with a step towards the denoised version
            alpha = (step + 1) / args.noise_steps  # Gradually increase clean signal
            current_embedding = alpha * denoised + (1 - alpha) * current_embedding

            # Add a small amount of noise to avoid collapse
            if step < args.noise_steps - 1:  # No noise in final step
                noise_scale = args.noise_level * (1 - alpha)
                current_embedding = (
                    current_embedding
                    + torch.randn_like(current_embedding) * noise_scale
                )

        # Final embedding
        generated_embedding = current_embedding

        # Compute similarity to input as a reference
        metrics = compute_metrics(generated_embedding, input_embedding)

        print(f"Generation Results:")
        print(f"Similarity to Input: {metrics['cosine_similarity']:.4f}")


def main():
    args = parse_args()

    # Load model and configuration
    model, config = load_model_and_config(args)

    # Initialize encoder
    print("Initializing encoder...")
    encoder = SonarEncoder(device=args.device)

    # Test on dataset if specified
    if args.test_data:
        test_on_dataset(model, encoder, args)

    # Run inference if input text is provided
    if args.input_text:
        run_inference(model, encoder, args)

    # If neither test data nor input text is provided
    if not args.test_data and not args.input_text:
        print(
            "No test data or input text provided. Please specify --test_data or --input_text"
        )
        print("Example usage:")
        print(
            "  Testing: python test.py --model_path saved_models/base_lcm_model_best.pth --test_data oscar --data_sample 100"
        )
        print(
            "  Inference: python test.py --model_path saved_models/base_lcm_model_best.pth --input_text 'This is a test sentence.'"
        )


if __name__ == "__main__":
    main()
