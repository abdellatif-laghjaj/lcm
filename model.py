import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Optional


class LCM(nn.Module):
    def __init__(
        self,
        encoder_model: str = "sentence-transformers/all-mpnet-base-v2",
        hidden_size: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Initialize sentence encoder
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)

        # Concept transformer layers
        self.concept_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Projection layers
        self.input_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_size)

        # Geometric regularization
        self.lambda_reg = 0.1

    def _get_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input_ids."""
        return (input_ids != self.tokenizer.pad_token_id).float()

    def _pool_embeddings(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool token embeddings to sentence embeddings using attention mask."""
        # Sum embeddings
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        summed = torch.sum(masked_embeddings, dim=1)

        # Get counts for averaging
        counts = torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1e-9)

        # Average
        pooled = summed / counts
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            labels: Optional labels for training [batch_size, seq_len]

        Returns:
            outputs: Model outputs
            loss: Optional loss if labels are provided
        """
        # Get attention mask if not provided
        if attention_mask is None:
            attention_mask = self._get_attention_mask(input_ids)

        # Get embeddings from encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        # Pool to sentence embeddings
        sentence_embeddings = self._pool_embeddings(
            encoder_outputs.last_hidden_state, attention_mask
        )

        # Project to concept space
        concepts = self.input_projection(sentence_embeddings)

        # Apply concept transformer
        transformed = self.concept_transformer(
            concepts, src_key_padding_mask=None  # No padding at concept level
        )

        # Project back to token space
        outputs = self.output_projection(transformed)
        outputs = self.norm(outputs)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Get target embeddings
            with torch.no_grad():
                target_outputs = self.encoder(
                    input_ids=labels,
                    attention_mask=self._get_attention_mask(labels),
                    return_dict=True,
                )
                target_embeddings = self._pool_embeddings(
                    target_outputs.last_hidden_state, self._get_attention_mask(labels)
                )

            # MSE loss
            loss = nn.functional.mse_loss(outputs, target_embeddings)

            # Add geometric regularization
            reg_loss = self.lambda_reg * torch.mean(
                torch.norm(outputs - sentence_embeddings, dim=-1)
            )
            loss = loss + reg_loss

        return (outputs, loss) if loss is not None else (outputs,)

    def encode(self, sentences: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode sentences to concept embeddings."""
        self.eval()
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                all_embeddings.append(outputs[0])

        return torch.cat(all_embeddings, dim=0)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        num_beams: int = 4,
        **kwargs
    ) -> torch.Tensor:
        """Generate text from concept embeddings."""
        # Get concepts
        concepts = self(input_ids, attention_mask)[0]

        # Project back to token space
        token_logits = self.output_projection(concepts)

        # Use beam search for generation
        outputs = []
        batch_size = token_logits.size(0)

        for i in range(batch_size):
            # Initialize beams with start token
            beams = [([], 0.0)]  # (tokens, score)

            for _ in range(max_length):
                candidates = []

                for beam_tokens, beam_score in beams:
                    if (
                        len(beam_tokens) > 0
                        and beam_tokens[-1] == self.tokenizer.eos_token_id
                    ):
                        candidates.append((beam_tokens, beam_score))
                        continue

                    # Get next token probabilities
                    logits = token_logits[i]  # Use concept embeddings
                    next_token_logits = torch.matmul(
                        logits, self.encoder.embeddings.word_embeddings.weight.t()
                    )

                    # Get top k tokens
                    values, indices = next_token_logits.topk(num_beams)

                    for value, token in zip(values, indices):
                        new_tokens = beam_tokens + [token.item()]
                        new_score = beam_score + value.item()
                        candidates.append((new_tokens, new_score))

                # Select top beams
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:num_beams]

                # Check if all beams ended
                if all(beam[0][-1] == self.tokenizer.eos_token_id for beam in beams):
                    break

            # Add best beam to outputs
            outputs.append(torch.tensor(beams[0][0]))

        return torch.stack(outputs)


class DiffusionLCM(nn.Module):
    """
    Implementation of a diffusion-based Large Concept Model (LCM) based on Meta's One-Tower architecture.
    This model uses a diffusion process to iteratively refine concept predictions.
    """

    def __init__(
        self,
        encoder_model: str = "sentence-transformers/all-mpnet-base-v2",
        hidden_size: int = 768,
        num_layers: int = 6,  # Increased from 4 to 6 for better performance
        num_heads: int = 8,
        dropout: float = 0.1,
        diffusion_steps: int = 10,
    ):
        super().__init__()

        # Initialize sentence encoder
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)

        # Concept transformer layers
        self.concept_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Projection layers
        self.input_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_size)

        # Diffusion parameters
        self.diffusion_steps = diffusion_steps
        self.betas = torch.linspace(0.0001, 0.02, diffusion_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)

        # Noise prediction network
        self.noise_pred_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        # Geometric regularization
        self.lambda_reg = 0.1

    def _get_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input_ids."""
        return (input_ids != self.tokenizer.pad_token_id).float()

    def _pool_embeddings(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool token embeddings to sentence embeddings using attention mask."""
        # Sum embeddings
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        summed = torch.sum(masked_embeddings, dim=1)

        # Get counts for averaging
        counts = torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1e-9)

        # Average
        pooled = summed / counts
        return pooled

    def add_noise(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to concept embeddings according to diffusion schedule."""
        # Time-dependent factor
        a = self.alphas_cumprod[t]
        if a.dim() == 0:
            a = a.unsqueeze(0)
        a = a.unsqueeze(1).to(x.device)  # Add dimensions for proper broadcasting

        # Sample noise
        noise = torch.randn_like(x)

        # Noisy sample: x_t = √α_t * x_0 + √(1 - α_t) * ϵ
        x_noisy = torch.sqrt(a) * x + torch.sqrt(1 - a) * noise

        return x_noisy, noise

    def predict_noise(
        self, x_noisy: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise in noisy concept embedding."""
        # Combine noisy concepts with context (input concepts)
        combined = torch.cat([x_noisy, context], dim=-1)

        # Predict noise
        predicted_noise = self.noise_pred_net(combined)

        return predicted_noise

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the diffusion-based LCM model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            labels: Optional labels for training [batch_size, seq_len]
            inference_steps: Number of diffusion steps for inference (defaults to self.diffusion_steps)

        Returns:
            outputs: Model outputs
            loss: Optional loss if labels are provided
        """
        # Get attention mask if not provided
        if attention_mask is None:
            attention_mask = self._get_attention_mask(input_ids)

        # Get embeddings from encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        # Pool to sentence embeddings
        sentence_embeddings = self._pool_embeddings(
            encoder_outputs.last_hidden_state, attention_mask
        )

        # Project to concept space
        context_concepts = self.input_projection(sentence_embeddings)

        # Apply concept transformer to get context representation
        context = self.concept_transformer(
            context_concepts, src_key_padding_mask=None  # No padding at concept level
        )

        # Calculate loss if labels provided (training mode)
        loss = None
        if labels is not None:
            # Get target embeddings
            with torch.no_grad():
                target_outputs = self.encoder(
                    input_ids=labels,
                    attention_mask=self._get_attention_mask(labels),
                    return_dict=True,
                )
                target_embeddings = self._pool_embeddings(
                    target_outputs.last_hidden_state, self._get_attention_mask(labels)
                )
                target_concepts = self.input_projection(target_embeddings)

            # Sample random timestep for each item in batch
            b = target_concepts.shape[0]
            t = torch.randint(
                0, self.diffusion_steps, (b,), device=target_concepts.device
            )

            # Add noise to target concepts
            noisy_concepts, noise = self.add_noise(target_concepts, t)

            # Predict noise
            predicted_noise = self.predict_noise(noisy_concepts, t, context)

            # Noise prediction loss (similar to DDPM)
            loss = nn.functional.mse_loss(predicted_noise, noise)

            # Add geometric regularization
            reg_loss = self.lambda_reg * torch.mean(
                torch.norm(target_concepts - context_concepts, dim=-1)
            )
            loss = loss + reg_loss

            # Return noisy concepts for visualization/debugging
            return (noisy_concepts, loss)

        # Inference mode: iterative denoising
        else:
            steps = inference_steps or self.diffusion_steps

            # Start with random noise
            x = torch.randn_like(context)

            # Iterative denoising
            for i in reversed(range(steps)):
                timestep = torch.ones(x.shape[0], device=x.device).long() * i

                # Predict noise
                predicted_noise = self.predict_noise(x, timestep, context)

                # Update sample with predicted noise (simplified DDIM-like step)
                alpha = self.alphas_cumprod[i]
                alpha_prev = self.alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0)

                sigma = 0.0  # Deterministic sampling

                # DDIM update step
                x = (
                    torch.sqrt(alpha_prev / alpha) * x
                    - torch.sqrt(1 - alpha_prev - sigma**2) * predicted_noise
                )

            # Final projection
            outputs = self.output_projection(x)
            outputs = self.norm(outputs)

            return (outputs,)

    def encode(self, sentences: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode sentences to concept embeddings."""
        self.eval()
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                all_embeddings.append(outputs[0])

        return torch.cat(all_embeddings, dim=0)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        num_beams: int = 4,
        diffusion_steps: int = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate text from concept embeddings using diffusion process."""
        # Get concepts through diffusion process
        inference_steps = (
            diffusion_steps or self.diffusion_steps // 2
        )  # Can use fewer steps for inference
        concepts = self(input_ids, attention_mask, inference_steps=inference_steps)[0]

        # Project back to token space
        token_logits = self.output_projection(concepts)

        # Use beam search for generation
        outputs = []
        batch_size = token_logits.size(0)

        for i in batch_size:
            # Initialize beams with start token
            beams = [([], 0.0)]  # (tokens, score)

            for _ in range(max_length):
                candidates = []

                for beam_tokens, beam_score in beams:
                    if (
                        len(beam_tokens) > 0
                        and beam_tokens[-1] == self.tokenizer.eos_token_id
                    ):
                        candidates.append((beam_tokens, beam_score))
                        continue

                    # Get next token probabilities
                    logits = token_logits[i]  # Use concept embeddings
                    next_token_logits = torch.matmul(
                        logits, self.encoder.embeddings.word_embeddings.weight.t()
                    )

                    # Get top k tokens
                    values, indices = next_token_logits.topk(num_beams)

                    for value, token in zip(values, indices):
                        new_tokens = beam_tokens + [token.item()]
                        new_score = beam_score + value.item()
                        candidates.append((new_tokens, new_score))

                # Select top beams
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:num_beams]

                # Check if all beams ended
                if all(beam[0][-1] == self.tokenizer.eos_token_id for beam in beams):
                    break

            # Add best beam to outputs
            outputs.append(torch.tensor(beams[0][0]))

        return torch.stack(outputs)


class TwoTowerDiffusionLCM(nn.Module):
    """
    Implementation of Meta's Two-Tower Diffusion Large Concept Model architecture.
    This model uses separate encoders for input and output concepts.
    """

    def __init__(
        self,
        encoder_model: str = "sentence-transformers/all-mpnet-base-v2",
        hidden_size: int = 768,
        num_layers_encoder: int = 4,
        num_layers_decoder: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        diffusion_steps: int = 10,
    ):
        super().__init__()

        # Initialize sentence encoder
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)

        # Input tower - processes input concepts
        self.input_tower = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers_encoder,
        )

        # Output tower - processes output concepts
        self.output_tower = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers_decoder,
        )

        # Projection layers
        self.input_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.cross_projection = nn.Linear(hidden_size, hidden_size)

        # Layer normalization
        self.norm_in = nn.LayerNorm(hidden_size)
        self.norm_out = nn.LayerNorm(hidden_size)

        # Diffusion parameters
        self.diffusion_steps = diffusion_steps
        self.betas = torch.linspace(0.0001, 0.02, diffusion_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)

        # Noise prediction network with cross-attention
        self.noise_pred_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Geometric regularization
        self.lambda_reg = 0.1

    def _get_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input_ids."""
        return (input_ids != self.tokenizer.pad_token_id).float()

    def _pool_embeddings(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool token embeddings to sentence embeddings using attention mask."""
        # Sum embeddings
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        summed = torch.sum(masked_embeddings, dim=1)

        # Get counts for averaging
        counts = torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1e-9)

        # Average
        pooled = summed / counts
        return pooled

    def add_noise(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to concept embeddings according to diffusion schedule."""
        # Time-dependent factor
        a = self.alphas_cumprod[t]
        if a.dim() == 0:
            a = a.unsqueeze(0)
        a = a.unsqueeze(1).to(x.device)  # Add dimensions for proper broadcasting

        # Sample noise
        noise = torch.randn_like(x)

        # Noisy sample: x_t = √α_t * x_0 + √(1 - α_t) * ϵ
        x_noisy = torch.sqrt(a) * x + torch.sqrt(1 - a) * noise

        return x_noisy, noise

    def predict_noise(
        self, x_noisy: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise in noisy concept embedding using cross-attention with context."""
        # Cross-attention between noisy concepts and context
        attn_output, _ = self.cross_attention(
            query=x_noisy.unsqueeze(1),
            key=context.unsqueeze(1),
            value=context.unsqueeze(1),
        )
        attn_output = attn_output.squeeze(1)

        # Combine with original noisy concepts
        combined = torch.cat([x_noisy, attn_output], dim=-1)

        # Predict noise
        predicted_noise = self.noise_pred_net(combined)

        return predicted_noise

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the Two-Tower diffusion-based LCM model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            labels: Optional labels for training [batch_size, seq_len]
            inference_steps: Number of diffusion steps for inference (defaults to self.diffusion_steps)

        Returns:
            outputs: Model outputs
            loss: Optional loss if labels are provided
        """
        # Get attention mask if not provided
        if attention_mask is None:
            attention_mask = self._get_attention_mask(input_ids)

        # Get embeddings from encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        # Pool to sentence embeddings
        sentence_embeddings = self._pool_embeddings(
            encoder_outputs.last_hidden_state, attention_mask
        )

        # Project to concept space
        input_concepts = self.input_projection(sentence_embeddings)

        # Process with input tower
        input_context = self.input_tower(
            input_concepts, src_key_padding_mask=None  # No padding at concept level
        )
        input_context = self.norm_in(input_context)

        # Calculate loss if labels provided (training mode)
        loss = None
        if labels is not None:
            # Get target embeddings
            with torch.no_grad():
                target_outputs = self.encoder(
                    input_ids=labels,
                    attention_mask=self._get_attention_mask(labels),
                    return_dict=True,
                )
                target_embeddings = self._pool_embeddings(
                    target_outputs.last_hidden_state, self._get_attention_mask(labels)
                )
                target_concepts = self.cross_projection(target_embeddings)

            # Sample random timestep for each item in batch
            b = target_concepts.shape[0]
            t = torch.randint(
                0, self.diffusion_steps, (b,), device=target_concepts.device
            )

            # Add noise to target concepts
            noisy_concepts, noise = self.add_noise(target_concepts, t)

            # Predict noise
            predicted_noise = self.predict_noise(noisy_concepts, t, input_context)

            # Noise prediction loss (similar to DDPM)
            loss = nn.functional.mse_loss(predicted_noise, noise)

            # Add geometric regularization
            reg_loss = self.lambda_reg * torch.mean(
                torch.norm(target_concepts - input_concepts, dim=-1)
            )
            loss = loss + reg_loss

            # Return noisy concepts for visualization/debugging
            return (noisy_concepts, loss)

        # Inference mode: iterative denoising
        else:
            steps = inference_steps or self.diffusion_steps

            # Start with random noise
            x = torch.randn_like(input_context)

            # Iterative denoising
            for i in reversed(range(steps)):
                timestep = torch.ones(x.shape[0], device=x.device).long() * i

                # Predict noise
                predicted_noise = self.predict_noise(x, timestep, input_context)

                # Update sample with predicted noise (simplified DDIM-like step)
                alpha = self.alphas_cumprod[i]
                alpha_prev = self.alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0)

                sigma = 0.0  # Deterministic sampling

                # DDIM update step
                x = (
                    torch.sqrt(alpha_prev / alpha) * x
                    - torch.sqrt(1 - alpha_prev - sigma**2) * predicted_noise
                )

            # Process with output tower
            x = self.output_tower(x)

            # Final projection
            outputs = self.output_projection(x)
            outputs = self.norm_out(outputs)

            return (outputs,)

    def encode(self, sentences: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode sentences to concept embeddings."""
        self.eval()
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                all_embeddings.append(outputs[0])

        return torch.cat(all_embeddings, dim=0)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        num_beams: int = 4,
        diffusion_steps: int = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate text from concept embeddings using two-tower diffusion process."""
        # Get concepts through two-tower diffusion process
        inference_steps = (
            diffusion_steps or self.diffusion_steps // 2
        )  # Can use fewer steps for inference
        concepts = self(input_ids, attention_mask, inference_steps=inference_steps)[0]

        # Project back to token space
        token_logits = self.output_projection(concepts)

        # Use beam search for generation
        outputs = []
        batch_size = token_logits.size(0)

        for i in range(batch_size):
            # Initialize beams with start token
            beams = [([], 0.0)]  # (tokens, score)

            for _ in range(max_length):
                candidates = []

                for beam_tokens, beam_score in beams:
                    if (
                        len(beam_tokens) > 0
                        and beam_tokens[-1] == self.tokenizer.eos_token_id
                    ):
                        candidates.append((beam_tokens, beam_score))
                        continue

                    # Get next token probabilities
                    logits = token_logits[i]  # Use concept embeddings
                    next_token_logits = torch.matmul(
                        logits, self.encoder.embeddings.word_embeddings.weight.t()
                    )

                    # Get top k tokens
                    values, indices = next_token_logits.topk(num_beams)

                    for value, token in zip(values, indices):
                        new_tokens = beam_tokens + [token.item()]
                        new_score = beam_score + value.item()
                        candidates.append((new_tokens, new_score))

                # Select top beams
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:num_beams]

                # Check if all beams ended
                if all(beam[0][-1] == self.tokenizer.eos_token_id for beam in beams):
                    break

            # Add best beam to outputs
            outputs.append(torch.tensor(beams[0][0]))

        return torch.stack(outputs)
