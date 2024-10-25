# Medical Image Classification using Deep Learning
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview
This project analyzes the effectiveness of deep learning models for medical image classification by comparing three architectures (EfficientNet-B4, Ensemble, and Vision Transformer) using a diabetic retinopathy dataset. Through detailed model comparison and performance analysis, we identify the most suitable architecture while considering technical limitations, ethical implications, and practical implementation challenges.

### Dataset Details:
- Source: [Kaggle/APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- Size: ~3,500 retinal images
- Classes: 5 severity levels
- Format: High-resolution fundus photographs

Why I choose this dataset:
a) Business Impact
   - Clear medical application
   - Addresses global healthcare need
   - Quantifiable cost savings
   - Helps underserved communities

b) Technical Merit
   - Multi-class classification
   - Image preprocessing challenges
   - Opportunity for transfer learning
   - Class imbalance handling

c) Model Variations:
   1. Basic CNN with data augmentation
   2. Transfer learning (ResNet50/VGG16)
   3. Attention-based model with class weighting

## Structure
```
├── data/
│   ├── raw/           # Original dataset
│   └── processed/     # Preprocessed images
├── models/
│   ├── efficientnet/
│   ├── ensemble/
│   └── vit/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── training/
│   └── evaluation/
├── docs/
└── results/
```

## Requirements
```
python>=3.8
torch>=1.9.0
tensorflow>=2.6.0
opencv-python>=4.5.3
numpy>=1.19.5
pandas>=1.3.0
```

## Installation
```bash
# Clone repository
git clone https://github.com/username/medical-image-classification.git
cd medical-image-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Preprocess data
python src/preprocessing/preprocess.py

# Train model
python src/training/train.py --model efficientnet

# Evaluate
python src/evaluation/evaluate.py --model efficientnet
```

## Models
| Model | Accuracy | Inference Time |
|-------|----------|----------------|
| EfficientNet-B4 | 89% | 0.8s |
| Ensemble | 92% | 1.5s |
| ViT | 91% | 1.2s |

## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/name`)
5. Open Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
