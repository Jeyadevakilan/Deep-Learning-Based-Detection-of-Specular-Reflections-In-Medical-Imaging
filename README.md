# Specular Reflection Detection and Enhancement
This project detects specular reflections in endoscopic images and enhances image clarity for better
visualization, highlighting the reflections instead of removing them. It uses a UNet architecture with
a semi-supervised learning approach to leverage both labeled and unlabeled data.
## Project Structure
```
├── data/ # Dataset classes
│ └── dataset.py # Dataset implementations
├── models/ # Model definitions
│ └── unet.py # UNet implementation
├── utils/ # Utility functions
│ └── visualization.py # Visualization utilities
├── enhanced_images/ # Directory for enhanced output images
├── pseudo_labels/ # Generated pseudo-labels from unlabeled data
├── runs/ # TensorBoard logs
├── train_initial.py # Initial training script with labeled data
├── train_semi_supervised.py # Semi-supervised training with pseudo-labels
├── inference.py # Interactive inference with Gradio UI
└── requirements.txt # Required packages
```

## Dataset Structure
The project assumes the following dataset structure:
```
D:/spec/Medical images.v1i.coco-segmentation/
├── train/ # Training images
│ └── masks/ # Training masks (with "_mask.png" suffix)
├── test/ # Testing images
│ └── masks/ # Testing masks (with "_mask.png" suffix)
└── unannotated/ # Unannotated images for semi-supervised learning
```

## Setup and Installation
1. Clone the repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Training Process
The training process consists of two phases:
### 1. Initial Training with Labeled Data
```bash
python train_initial.py
```

This script trains the model on the labeled data (200+ annotated images). It saves the best model to
`models/initial_model.pth`.


### 2. Semi-Supervised Training with Unlabeled Data
```bash
python train_semi_supervised.py
```

This script:
1. Loads the initial model
2. Generates pseudo-labels for unlabeled data (900+ unannotated images)
3. Trains a model using both the labeled data and the pseudo-labeled data
4. Performs final fine-tuning
5. Saves the best model to `models/final_model.pth`
## Inference and Visualization
To run the interactive inference UI:
```bash
python inference.py
```

This launches a Gradio web interface where you can:
1. Upload an endoscopic image
2. View the detected reflections
3. See the enhanced image with highlighted reflections
4. All processed images are saved to the `enhanced_images/` directory
## Model Architecture
The model uses a UNet architecture with a ResNet34 encoder pre-trained on ImageNet for feature extraction.
This provides better generalization and faster convergence.
## Performance Metrics
The model's performance is measured using the Intersection over Union (IoU) metric, which quantifies
how well the predicted reflection masks match the ground truth.
## Visualization Examples
The inference script provides four visualization outputs:
- Original image
- Reflection detection mask
- Overlay visualization (original with highlighted reflections)
- Enhanced image with improved clarity and highlighted reflections
## Training Tips
- For best results, use a GPU-enabled system
- Monitor training progress with TensorBoard: `tensorboard --logdir=runs`
- The semi-supervised approach significantly improves performance by leveraging unlabeled data
## Requirements
- Python 3.7+
- PyTorch 1.8+
- CUDA-compatible GPU (recommended)
- See requirements.txt for complete list