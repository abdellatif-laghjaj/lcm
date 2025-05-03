import torch
import faiss
import nltk
import os
import argparse
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from model import LCM, DiffusionLCM, TwoTowerDiffusionLCM


class ConceptRetriever:
    def __init__(
        self,
        model_path: str = "base_model.pt",
        model_type: str = "base",
        encoder_model: str = "sentence-transformers/all-mpnet-base-v2",
        diffusion_steps: int = 10,
        batch_size: int = 8,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate summaries using Large Concept Models"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="base_model.pt",
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
        default="retrieval",
        choices=["retrieval", "generation"],
        help="Method to use for generating summaries",
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
    args = parser.parse_args()

    # Initialize model
    retriever = ConceptRetriever(
        model_path=args.model_path,
        model_type=args.model_type,
        encoder_model=args.encoder_model,
        diffusion_steps=args.diffusion_steps,
    )

    # Example articles
    articles = [
        "The quick brown fox jumps over the lazy dog. This is a test sentence. Another test sentence here.",
        "Different article with some content. More sentences here. Final test sentence.",
    ]

    if args.method == "retrieval":
        # Build index
        retriever.build_index(articles)

        # Generate summary using retrieval
        test_article = "Test article about a fox and a dog."
        summary = retriever.generate_summary(test_article, temperature=args.temperature)
        print(f"\nGenerated summary (retrieval): {summary}")

    else:  # generation
        # Generate summary using direct generation
        test_article = "Test article about a fox and a dog."
        summary = retriever.direct_generation(
            test_article,
            diffusion_steps=(
                args.diffusion_steps // 2 if args.model_type != "base" else None
            ),
        )
        print(f"\nGenerated summary (direct): {summary}")


if __name__ == "__main__":
    main()
