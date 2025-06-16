# Image Captioning with Transformer Architecture

This project implements an advanced image captioning system using a Transformer-based architecture. The model is trained on the Flickr8k dataset and can generate natural language descriptions for images.

## Features

- Transformer-based architecture for image captioning
- Pre-trained ResNet50 for image feature extraction
- Beam search for improved caption generation
- Support for both training and inference modes
- Early stopping and model checkpointing
- Focal Loss with label smoothing for better training
- Mixed precision training support
- Test-time augmentation for improved inference

## Project Structure

```
.
├── Image Captioning.ipynb     # Main notebook containing the implementation
├── caption_model.pth         # Trained model weights
├── best_caption_model.pth    # Best model weights during training
├── image_features.pkl        # Cached image features
├── Flickr8k_Dataset/         # Image dataset directory
└── Flickr8k_text/           # Text dataset directory
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- PIL
- numpy
- pandas
- nltk
- scikit-learn

You can install all required dependencies using:
```bash
pip install numpy pandas pillow matplotlib torch torchvision scikit-learn nltk
```

Note: Some of these packages (like `torch` and `torchvision`) might require specific versions depending on your CUDA setup. Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) for the appropriate installation command for your system.

## Dataset

The project uses the Flickr8k dataset, which consists of:
- 8,000 images
- 5 captions per image
- Training, validation, and test splits

## Resources

### Google Drive Resources
The following resources are available on Google Drive:
- [Trained Models and Datasets](YOUR_GOOGLE_DRIVE_LINK_HERE)
  - `caption_model.pth`: Standard trained model
  - `best_caption_model.pth`: Best performing model
  - `image_features.pkl`: Pre-extracted image features
  - `Flickr8k_Dataset.zip`: Image dataset
  - `Flickr8k_text.zip`: Text dataset with captions

To use these resources:
1. Download the required files from the Google Drive link
2. Extract the dataset files to their respective directories
3. Place the model files in the project root directory


## Model Architecture

The image captioning model consists of several key components:

1. **Feature Extractor**: Uses ResNet50 to extract image features
2. **Transformer Decoder**: Generates captions using a transformer architecture
3. **Cross-Attention**: Multiple cross-attention layers for better image-text alignment
4. **Beam Search**: Implements beam search for improved caption generation

## Training

To train the model:

1. Ensure you have the Flickr8k dataset downloaded and extracted
2. Run the notebook and select Option 1
3. Follow the prompts to specify:
   - Text directory path
   - Image directory path
   - Number of training epochs
4. The model will train with the following features:
   - Early stopping
   - Model checkpointing
   - Learning rate scheduling
   - Mixed precision training (if available)
   - Gradient clipping

## Generating Captions

To generate captions for new images:

1. Run the notebook and select Option 2
2. Choose to caption a single image
3. Provide the path to the image
4. The model will generate a caption using beam search

## Model Features

- **Vocabulary Building**: Creates a vocabulary from the training captions
- **Data Preprocessing**: Handles image resizing, augmentation, and caption cleaning
- **Feature Extraction**: Extracts and caches image features for faster training
- **Beam Search**: Implements beam search for improved caption generation
- **Test-Time Augmentation**: Uses multiple augmentations during inference for better results

## Performance Optimizations

- Mixed precision training for faster training on supported hardware
- Cached image features to avoid redundant computation
- Efficient data loading with pinned memory
- Gradient clipping to prevent exploding gradients
- Early stopping to prevent overfitting

## Usage Example

```python
# Load the model
model, vocab_to_idx, idx_to_vocab, vocab_size = load_model_if_exists('caption_model.pth')

# Initialize feature extractor
feature_extractor = FeatureExtractor()

# Generate caption for an image
caption = generate_caption(model, 'path/to/image.jpg', feature_extractor, idx_to_vocab)
print(caption)
```

## Training Parameters

- Batch size: 32
- Learning rate: 2e-4
- Embedding dimension: 512
- Number of transformer layers: 6
- Number of attention heads: 8
- Dropout rate: 0.1
- Maximum caption length: 22

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flickr8k dataset creators
- PyTorch team
- Transformer paper authors 