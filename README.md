# Image Captioning System

This project implements a deep learning image captioning model using a Transformer-based architecture. The model is trained on the Flickr8k dataset and generates natural language descriptions for images.

The system combines ResNet50 for image feature extraction with a Transformer decoder to generate contextually relevant captions for input images.

## Features

- Transformer-based architecture for image captioning
- Pre-trained ResNet50 for image feature extraction
- Beam search for improved caption generation
- Gradient clipping to prevent exploding gradients
- Vocabulary building from the training captions
- Support for both training and inference modes
- Early stopping and model checkpointing
- Focal Loss with label smoothing for better training

## Project Structure

```
├── Image Captioning.ipynb    # Main notebook containing the full end-to-end implementation
├── caption_model.pth         # Trained model
├── best_caption_model.pth    # Best model during training
├── image_features.pkl        # Cached image features
├── Flickr8k_Dataset/         # Image dataset directory
├── Flickr8k_text/            # Text dataset directory
├── ... images and other files ...
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

You can install all the required dependencies using:
```bash
pip install numpy pandas pillow matplotlib torch torchvision scikit-learn nltk
```

## Resources

The following resources are provided using a Google Drive link:
- [Trained Models and Datasets](https://drive.google.com/drive/folders/16QP-v5A3NzAx67PHGeNyDU3qQgt_nfeU?usp=sharing): 
  - `caption_model.pth`: Standard trained model
  - `best_caption_model.pth`: Best performing model
  - `image_features.pkl`: Extracted image features
  - `Flickr8k_Dataset.zip`: Image dataset
  - `Flickr8k_text.zip`: Text dataset with captions

To use these resources:
1. Download the required files from the Google Drive link.
2. Extract the dataset files, and also place the model files, in the main cloned project directory.

## Training

To train the model:
1. Ensure you have the Flickr8k dataset downloaded and extracted in the main project directory.
2. Run the notebook and select Option 1.
3. Follow the prompts to specify the path deatils and the number of training epochs (or just press 'Enter' to move forward with the default settings).
4. The model will train with the following features:
   - Early stopping
   - Model checkpointing
   - Learning rate scheduling
   - Gradient clipping

## Generating Captions

To generate captions for new images:

1. Run the notebook and select Option 2 (you can directly run this without training through Option 1 if you have downloaded the models already from the provided Google Drive link).
2. Choose to caption a single image, and then provide a path to the required image.
3. The model will generate a caption for the image.

## Training Parameters

- Batch size: 32
- Learning rate: Dynamic
- Embedding dimension: 512
- Number of transformer layers: 6
- Number of attention heads: 8
- Dropout rate: 0.1
- Maximum caption length: 22

## Contributing

Contributions are welcome!

## License

Distributed under the MIT License.  
