import torch
import argparse
import spacy
from tqdm.auto import tqdm
import numpy as np

from baselcm import BaseLCM, SonarEncoder
from utils import add_noise_to_embeddings  # Assuming add_noise is in utils

# Set random seed for reproducibility if needed
# torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate embeddings using a trained BaseLCM model"
    )
    # Model & Data Args
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved BaseLCM model state_dict (.pth file)",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        required=True,
        help="Initial text sequence to start generation/testing.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code for Sonar encoding (e.g., 'en')",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=10,
        help="Sequence length used during training",
    )  # Must match training

    # Generation Args
    parser.add_argument(
        "--mode",
        type=str,
        choices=["continuation", "denoising"],
        default="continuation",
        help="Generation mode",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5,
        help="Number of steps to generate (for continuation mode)",
    )
    parser.add_argument(
        "--noise_level", type=float, default=0.05, help="Noise level for denoising mode"
    )

    # Model Config (must match the trained model)
    parser.add_argument(
        "--input_dim",
        type=int,
        default=1024,
        help="Input dimension (Sonar embedding size)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=1024,
        help="Hidden dimension within the Transformer",
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=1024,
        help="Output dimension (should match input_dim)",
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=12,
        help="Number of Transformer Decoder layers",
    )
    parser.add_argument(
        "--ff_dim",
        type=int,
        default=4096,
        help="Feedforward dimension in Transformer layers",
    )
    parser.add_argument(
        "--max_seq_len_model",
        type=int,
        default=50,
        help="Maximum sequence length the model's positional encoding can handle",
    )

    # Misc
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device ('cuda', 'cpu'). Auto-detects if None.",
    )
    parser.add_argument(
        "--encoding_batch_size",
        type=int,
        default=32,
        help="Batch size for sentence encoding",
    )

    return parser.parse_args()


# Centralized device management
def get_device(args_device: str | None) -> torch.device:
    if args_device:
        return torch.device(args_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper to load spacy model safely
def load_spacy_model(model_name="en_core_web_sm"):
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Spacy model '{model_name}' not found. Downloading...")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
    return nlp


def generate(args):
    device = get_device(args.device)
    print(f"Using device: {device}")

    # --- 1. Load Model ---
    print(f"Loading model from {args.model_path}")
    model = BaseLCM(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        output_dim=args.output_dim,
        max_seq_len=args.max_seq_len_model,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully.")

    # --- 2. Prepare Input ---
    print("Loading spacy for sentence splitting...")
    nlp = load_spacy_model()

    print("Processing input text...")
    doc = nlp(args.input_text)
    input_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if len(input_sentences) < 1:
        print("Error: Input text did not yield any sentences.")
        return
    if len(input_sentences) < args.sequence_length and args.mode == "continuation":
        print(
            f"Warning: Input text has {len(input_sentences)} sentences, less than sequence_length {args.sequence_length}. Using all available sentences as initial context."
        )
        current_seq_len = len(input_sentences)
    else:
        # Use the last `sequence_length` sentences if more are provided
        input_sentences = input_sentences[-args.sequence_length :]
        current_seq_len = args.sequence_length

    print(f"Using initial sentences: {input_sentences}")

    print("Initializing SonarEncoder...")
    encoder = SonarEncoder(device=str(device))

    print("Encoding initial sentences...")
    initial_embeddings = encoder.encode(
        input_sentences, lang=args.lang, batch_size=args.encoding_batch_size
    ).to(
        device
    )  # Keep embeddings on the target device

    if initial_embeddings.shape[0] == 0:
        print("Error: Failed to encode initial sentences.")
        return

    # --- 3. Perform Generation/Testing ---
    generated_embeddings_list = [initial_embeddings]

    with torch.inference_mode():
        if args.mode == "continuation":
            print(f"Starting continuation generation for {args.num_steps} steps...")
            current_sequence = initial_embeddings

            for step in tqdm(range(args.num_steps), desc="Generating Steps"):
                # Ensure input sequence has the correct shape (batch=1, seq_len, dim)
                # If the current sequence is shorter than required (initial phase), use it as is.
                # Once enough steps are generated, always use the last `sequence_length` embeddings.
                if current_sequence.shape[0] > args.sequence_length:
                    input_seq_tensor = current_sequence[
                        -args.sequence_length :
                    ].unsqueeze(0)
                else:
                    input_seq_tensor = current_sequence.unsqueeze(0)

                # Predict the embedding for the next step
                predicted_output_seq = model(input_seq_tensor)
                next_embedding = predicted_output_seq[
                    :, -1, :
                ]  # Shape: (1, output_dim)

                # Append the predicted embedding (remove batch dim)
                generated_embeddings_list.append(next_embedding.cpu())  # Store on CPU
                current_sequence = torch.cat(
                    [current_sequence, next_embedding.to(current_sequence.device)],
                    dim=0,
                )

            print("Continuation generation complete.")

        elif args.mode == "denoising":
            print("Performing denoising test...")
            if initial_embeddings.shape[0] < args.sequence_length:
                print(
                    f"Error: Denoising mode requires at least {args.sequence_length} initial sentences for context. Got {initial_embeddings.shape[0]}."
                )
                return

            # Take the required sequence length
            context_sequence = initial_embeddings[-args.sequence_length :]

            # Predict next step from clean context
            clean_input_tensor = context_sequence.unsqueeze(0)  # Add batch dim
            clean_pred_output = model(clean_input_tensor)
            clean_next_embedding = clean_pred_output[:, -1, :].cpu()  # Store on CPU

            # Predict next step from noisy context
            noisy_context = add_noise_to_embeddings(context_sequence, args.noise_level)
            noisy_input_tensor = noisy_context.unsqueeze(0)  # Add batch dim
            noisy_pred_output = model(noisy_input_tensor)
            noisy_next_embedding = noisy_pred_output[:, -1, :].cpu()  # Store on CPU

            print("Denoising test complete.")
            print(
                "Predicted embedding from CLEAN context (first 5 dims):",
                clean_next_embedding.flatten()[:5].numpy(),
            )
            print(
                "Predicted embedding from NOISY context (first 5 dims):",
                noisy_next_embedding.flatten()[:5].numpy(),
            )
            # Compare the two predictions
            similarity = torch.nn.functional.cosine_similarity(
                clean_next_embedding, noisy_next_embedding, dim=-1
            )
            print(
                f"Cosine Similarity between clean/noisy predictions: {similarity.item():.4f}"
            )
            # We only have the predictions, not the ground truth next step here.
            generated_embeddings_list = [
                context_sequence.cpu(),
                clean_next_embedding,
                noisy_next_embedding,
            ]

    # --- 4. Output Results ---
    print("\n--- Results ---")
    print(f"Mode: {args.mode}")

    if args.mode == "continuation":
        print(f"Generated {args.num_steps} additional concept embeddings.")
        print("Initial context embeddings shape:", generated_embeddings_list[0].shape)
        for i in range(1, len(generated_embeddings_list)):
            print(
                f"Generated Step {i} embedding shape: {generated_embeddings_list[i].shape}"
            )
            print(
                f"  Embedding (first 5 dims): {generated_embeddings_list[i].flatten()[:5].numpy()}"
            )
        # Combine all generated embeddings into a single tensor for potential saving/analysis
        final_embedding_sequence = torch.cat(generated_embeddings_list, dim=0)
        print("Full generated sequence shape:", final_embedding_sequence.shape)
        # np.save("generated_embeddings.npy", final_embedding_sequence.numpy()) # Optional: save
        # print("Saved generated embeddings to generated_embeddings.npy")

    elif args.mode == "denoising":
        print("Context sequence shape:", generated_embeddings_list[0].shape)
        print("Clean prediction shape:", generated_embeddings_list[1].shape)
        print("Noisy prediction shape:", generated_embeddings_list[2].shape)

    print(
        "\nNOTE: Output consists of embeddings. Decoding requires the corresponding Sonar Decoder model."
    )


if __name__ == "__main__":
    args = parse_args()
    generate(args)
