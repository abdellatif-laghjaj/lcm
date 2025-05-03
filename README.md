# Large Concept Model (LCM) - Practical Implementation

This repository contains the implementation of my technical report paper "Towards Practical Concept-Based Language Models: An Efficiency-Focused Implementation". Our work demonstrates significant efficiency improvements in language processing through concept-based approaches.

## Key Features

- 🚀 3.8× faster inference through sentence-level processing
- 📉 Linear memory scaling (O(n)) for long sequences
- 🌍 Multilingual support with minimal performance drop
- 💡 Adaptive concept quantization
- 🔄 Hybrid attention mechanism
- 📊 Geometric regularization for semantic fidelity
- ⚡ Diffusion-based concept refinement (NEW)
- 🏗️ Two-Tower architecture for improved performance (NEW)

## Installation

```bash
# Clone the repository
git clone https://github.com/arimanyus/large-concept-model
cd large-concept-model

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from model import LCM, DiffusionLCM, TwoTowerDiffusionLCM

# Initialize base model
base_model = LCM.from_pretrained('lcm-base')

# Or use the more advanced diffusion model
diffusion_model = DiffusionLCM(diffusion_steps=10)

# Process text
concepts = model.extract_concepts("Your input text here")
output = model.generate(concepts)
```

## Model Architectures

This repository implements three different Large Concept Model architectures:

### 1. Base-LCM

Our original implementation that processes concepts in a transformer decoder. It works by:

- Converting text to sentence embeddings (concepts)
- Processing these concepts with a transformer
- Projecting concepts back to text space

### 2. Diffusion LCM (One-Tower)

Based on Meta's One-Tower architecture, this diffusion-based model refines concept predictions iteratively:

- Starts with a noisy concept representation
- Uses diffusion process to iteratively denoise the concept
- Achieves better performance by considering multiple plausible outputs

```bash
python train.py --model_type diffusion --diffusion_steps 10
```

### 3. Two-Tower Diffusion LCM

Based on Meta's Two-Tower architecture, this model uses separate transformers for input and output:

- First tower processes input concepts
- Second tower processes output concepts
- Cross-attention mechanism connects the two towers
- Better separation of encoder and decoder functionality

```bash
python train.py --model_type two_tower --diffusion_steps 10
```

## Training

To train your own model:

```bash
python train.py \
    --model_type base \           # Options: base, diffusion, two_tower
    --batch_size 32 \
    --learning_rate 5e-5 \
    --epochs 5 \
    --encoder_model sentence-transformers/all-mpnet-base-v2 \
    --dataset cnn_dailymail \
    --dataset_version 3.0.0
```

For diffusion-based models, you can also specify:

```bash
python train.py \
    --model_type diffusion \      # or two_tower
    --diffusion_steps 10 \        # Number of diffusion steps
    --batch_size 16               # Smaller batch size due to increased memory usage
```

## Generation

Generate summaries using the trained models:

```bash
python generate.py \
    --model_path base_model.pt \           # Path to trained model
    --model_type base \                    # Options: base, diffusion, two_tower
    --method retrieval \                   # Options: retrieval, generation
    --temperature 1.0                      # For retrieval sampling
```

For diffusion-based generation:

```bash
python generate.py \
    --model_type diffusion \               # or two_tower
    --method generation \
    --diffusion_steps 10                   # More steps = higher quality, slower generation
```

## Evaluation

Run evaluation on standard benchmarks:

```bash
python evaluate.py \
    --model_path path/to/model \
    --dataset cnn_dailymail
```

## Model Architecture

Our implementation consists of three main components:

1. **Concept Formation**: Converts text to compressed concept embeddings
2. **Concept Processing**:
   - Base-LCM: 4-layer transformer
   - Diffusion LCM: 6-layer transformer with diffusion process
   - Two-Tower: Separate encoder and decoder transformers
3. **Hybrid Generation**: Combines concept and token-level processing

## Hyperparameters

Key hyperparameters used in our experiments:

| Parameter            | Base | Diffusion | Two-Tower |
| -------------------- | ---- | --------- | --------- |
| Learning Rate        | 5e-5 | 5e-5      | 5e-5      |
| Batch Size           | 32   | 16        | 16        |
| Transformer Layers   | 4    | 6         | 4+4       |
| Attention Heads      | 8    | 8         | 8         |
| Diffusion Steps      | -    | 10        | 10        |
| α (Hybrid Attention) | 0.7  | 0.7       | 0.7       |

## Results

Our models achieve:

- Base-LCM: 82% ROUGE-L retention compared to BART
- Diffusion LCM: 87% ROUGE-L retention
- Two-Tower LCM: 89% ROUGE-L retention
- All models show excellent multilingual performance with only 4-8% average performance drop

## Visualization

Generate concept space visualizations:

```bash
python visualize.py --embedding_dir path/to/embeddings
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{tiwari2024towards,
  title={Towards Practical Concept-Based Language Models: An Efficiency-Focused Implementation},
  author={Tiwari, Vivek K.},
  journal={arXiv preprint arXiv:2024.6154975},
  year={2024}
}
```

## Acknowledgments

- The authors of the original LCM paper from Meta AI
- IBM for technical guidance
- The open-source NLP community

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please check our [contributing guidelines](CONTRIBUTING.md) for details.

## Contact

- Vivek K. Tiwari - vivek.tiwari4@ibm.com / vivek3312@gmail.com
- Project Link: https://github.com/arimanyus/large-concept-model
