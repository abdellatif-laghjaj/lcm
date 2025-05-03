import torch
import faiss
import nltk
import os
import argparse
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
from model import LCM, DiffusionLCM, TwoTowerDiffusionLCM
from datasets import load_dataset
from rouge_score import rouge_scorer
import pandas as pd
import matplotlib.pyplot as plt


class ConceptRetriever:
    def __init__(
        self,
        model_path: str = "base_model.pt",
        model_type: str = "base",
        encoder_model: str = "sentence-transformers/all-mpnet-base-v2",
        diffusion_steps: int = 10,
        batch_size: int = 8,
        device: str = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size
        self.model_type = model_type

        # Load the model
        print(
            f"Loading {model_type.upper()}-LCM model from {os.path.abspath(model_path)}"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Initialize the appropriate model architecture
        if model_type == "base":
            self.model = LCM(encoder_model=encoder_model)
        elif model_type == "diffusion":
            self.model = DiffusionLCM(
                encoder_model=encoder_model, diffusion_steps=diffusion_steps
            )
        elif model_type == "two_tower":
            self.model = TwoTowerDiffusionLCM(
                encoder_model=encoder_model, diffusion_steps=diffusion_steps
            )
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. Must be one of 'base', 'diffusion', or 'two_tower'"
            )

        # Load model weights
        try:
            # Try loading state dict directly first
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except:
            # If that fails, try loading from a checkpoint dictionary
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                raise ValueError(f"Could not load model from {model_path}")

        self.model.to(self.device)
        self.model.eval()

        # Initialize FAISS index
        self.index = None
        self.sentences = []

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)

        # Download punkt if needed
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

    def _get_embeddings(self, sentences: List[str]) -> torch.Tensor:
        """Get embeddings for a batch of sentences."""
        try:
            # Process in batches
            all_embeddings = []
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i : i + self.batch_size]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs[0]
                    all_embeddings.append(embeddings.cpu())

            return torch.cat(all_embeddings, dim=0)
        except Exception as e:
            print(f"Error getting embeddings: {str(e)}")
            return None

    def build_index(self, articles: List[str]):
        """Build FAISS index from articles."""
        try:
            print("Building index...")
            all_sentences = []
            for article in tqdm(articles):
                sentences = nltk.sent_tokenize(article)
                all_sentences.extend(sentences)

            if not all_sentences:
                print("Warning: No sentences found in articles")
                return

            self.sentences = all_sentences
            embeddings = self._get_embeddings(all_sentences)

            if embeddings is None or embeddings.size(0) == 0:
                print("Error: No valid embeddings generated")
                return

            # Convert to numpy and normalize
            embeddings_np = embeddings.numpy()
            faiss.normalize_L2(embeddings_np)

            # Build index
            self.index = faiss.IndexFlatIP(embeddings_np.shape[1])
            self.index.add(embeddings_np)
            print(f"Index built with {len(all_sentences)} sentences")

        except Exception as e:
            print(f"Error building index: {str(e)}")

    def generate_summary(
        self, article: str, num_sentences: int = 3, temperature: float = 1.0
    ) -> str:
        """Generate summary by retrieving similar sentences."""
        try:
            if self.index is None:
                raise ValueError("Index not built. Call build_index first.")

            # Get query embedding
            query_embedding = self._get_embeddings([article])
            if query_embedding is None:
                return ""

            # Convert to numpy and normalize
            query_np = query_embedding.numpy()
            faiss.normalize_L2(query_np)

            # Search index
            D, I = self.index.search(query_np, num_sentences)

            # Add temperature sampling (if temperature > 0)
            if temperature > 0:
                # Softmax with temperature
                scores = np.exp(D[0] / temperature)
                scores = scores / np.sum(scores)

                # Sample according to probabilities
                selected_indices = np.random.choice(
                    len(scores),
                    size=min(num_sentences, len(scores)),
                    replace=False,
                    p=scores,
                )

                # Get selected sentences
                summary_sentences = [self.sentences[I[0][i]] for i in selected_indices]
            else:
                # Get top sentences
                summary_sentences = [self.sentences[i] for i in I[0]]

            return " ".join(summary_sentences)

        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return ""

    def direct_generation(
        self,
        article: str,
        max_length: int = 128,
        num_beams: int = 4,
        diffusion_steps: int = None,
    ) -> str:
        """Generate text directly using the model's generation capabilities."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                article, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate output
            with torch.no_grad():
                if self.model_type == "base":
                    output_ids = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=max_length,
                        num_beams=num_beams,
                    )
                else:
                    # For diffusion models, optionally specify diffusion steps
                    output_ids = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=max_length,
                        num_beams=num_beams,
                        diffusion_steps=diffusion_steps,
                    )

            # Decode output
            generated_text = self.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
            return generated_text

        except Exception as e:
            print(f"Error in direct generation: {str(e)}")
            return ""

    def evaluate(
        self, dataset, num_samples=10, method="generation", diffusion_steps=None
    ):
        """Evaluate the model on a dataset and compute ROUGE scores."""
        if num_samples > len(dataset):
            num_samples = len(dataset)

        # Sample a subset of data for evaluation
        indices = np.random.choice(len(dataset), num_samples, replace=False)

        rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        results = []

        for i in tqdm(indices, desc="Evaluating"):
            example = dataset[i]
            article = example["article"]
            reference = example["highlights"]

            if method == "retrieval":
                # Build index from the article
                self.build_index([article])
                prediction = self.generate_summary(article, num_sentences=3)
            else:
                # Direct generation
                prediction = self.direct_generation(
                    article, max_length=128, diffusion_steps=diffusion_steps
                )

            # Calculate ROUGE scores
            scores = rouge.score(reference, prediction)

            results.append(
                {
                    "article": article[:100]
                    + "...",  # Truncate long articles for display
                    "reference": reference,
                    "prediction": prediction,
                    "rouge1": scores["rouge1"].fmeasure,
                    "rouge2": scores["rouge2"].fmeasure,
                    "rougeL": scores["rougeL"].fmeasure,
                }
            )

        # Calculate average scores
        avg_rouge1 = np.mean([r["rouge1"] for r in results])
        avg_rouge2 = np.mean([r["rouge2"] for r in results])
        avg_rougeL = np.mean([r["rougeL"] for r in results])

        print(f"\nAverage ROUGE Scores ({method}):")
        print(f"ROUGE-1: {avg_rouge1:.4f}")
        print(f"ROUGE-2: {avg_rouge2:.4f}")
        print(f"ROUGE-L: {avg_rougeL:.4f}")

        return results

    def visualize_results(self, results, method="generation"):
        """Visualize the evaluation results."""
        # Convert results to DataFrame for easier manipulation
        df = pd.DataFrame(results)

        # Create bar chart of ROUGE scores
        plt.figure(figsize=(10, 6))
        avg_scores = [df["rouge1"].mean(), df["rouge2"].mean(), df["rougeL"].mean()]
        plt.bar(
            ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
            avg_scores,
            color=["blue", "green", "red"],
        )
        plt.title(f"Average ROUGE Scores ({method.capitalize()} Method)")
        plt.ylabel("F-measure")
        plt.ylim(0, 1)

        # Add score values on top of bars
        for i, score in enumerate(avg_scores):
            plt.text(i, score + 0.05, f"{score:.4f}", ha="center")

        plt.tight_layout()
        plt.savefig(f"{self.model_type}_lcm_{method}_scores.png")
        print(
            f"Scores visualization saved to {self.model_type}_lcm_{method}_scores.png"
        )

        # Create a summary table
        print("\nSample Results:")
        for i in range(min(3, len(results))):
            print(f"\nExample {i+1}:")
            print(f"Reference: {results[i]['reference']}")
            print(f"Prediction: {results[i]['prediction']}")
            print(
                f"ROUGE-1: {results[i]['rouge1']:.4f}, ROUGE-2: {results[i]['rouge2']:.4f}, ROUGE-L: {results[i]['rougeL']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generate summaries using Large Concept Models"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/base_model.pt",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="base",
        choices=["base", "diffusion", "two_tower"],
        help="Type of LCM model to use",
    )
    parser.add_argument(
        "--encoder_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Encoder model to use",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="generation",
        choices=["retrieval", "generation", "evaluate"],
        help="Method to use for generating summaries or evaluation",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=10,
        help="Number of diffusion steps (for diffusion models)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for retrieval sampling",
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda, cpu)"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize evaluation results"
    )
    args = parser.parse_args()

    # Initialize model
    retriever = ConceptRetriever(
        model_path=args.model_path,
        model_type=args.model_type,
        encoder_model=args.encoder_model,
        diffusion_steps=args.diffusion_steps,
        device=args.device,
    )

    # Example articles for simple testing
    news_articles = [
        "Scientists have discovered a new species of deep-sea fish that can survive at extreme depths. The fish has special adaptations including pressure-resistant cells and unique enzyme systems. Researchers believe this discovery could help develop new medical treatments for high blood pressure and other conditions.",
        "The city council voted yesterday to approve the new urban development plan. The plan includes affordable housing units, green spaces, and improved public transportation options. Local residents expressed mixed reactions, with some praising the focus on sustainability while others expressed concerns about potential increases in traffic congestion.",
    ]

    if args.method == "retrieval":
        # Build index
        retriever.build_index(news_articles)

        # Generate summary using retrieval
        test_article = "A team of marine biologists has found new fish species in the deep ocean trenches. These fish have unusual adaptations."
        summary = retriever.generate_summary(test_article, temperature=args.temperature)
        print(f"\nGenerated summary (retrieval): {summary}")

    elif args.method == "generation":
        # Generate summary using direct generation
        test_article = "A team of marine biologists has found new fish species in the deep ocean trenches. These fish have unusual adaptations."
        summary = retriever.direct_generation(
            test_article,
            diffusion_steps=(
                args.diffusion_steps // 2 if args.model_type != "base" else None
            ),
        )
        print(f"\nGenerated summary (direct): {summary}")

    elif args.method == "evaluate":
        # Load dataset for evaluation
        print("Loading dataset for evaluation...")
        dataset = load_dataset("cnn_dailymail", "3.0.0")["test"]

        # Run evaluation
        print(
            f"Evaluating model using {args.method} method on {args.num_samples} samples..."
        )
        results = retriever.evaluate(
            dataset,
            num_samples=args.num_samples,
            method="generation",
            diffusion_steps=(
                args.diffusion_steps // 2 if args.model_type != "base" else None
            ),
        )

        # Visualize results if requested
        if args.visualize:
            try:
                retriever.visualize_results(results, method="generation")
            except Exception as e:
                print(f"Error visualizing results: {str(e)}")


if __name__ == "__main__":
    main()
