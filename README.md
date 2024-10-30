# Medical Image Classification using Deep Learning


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

## 🎯 Key Features

- Transfer learning with EfficientNetB0 architecture
- K-fold cross-validation implementation
- Advanced image preprocessing and augmentation
- Automated severity classification (0-4 scale)
- Performance metrics and visualization


## Models
| Model | Accuracy | Inference Time |
|-------|----------|----------------|
| EfficientNet-B4 | 89% | 0.8s |
| Ensemble | 92% | 1.5s |
| ViT | 91% | 1.2s |

## References
* APTOS 2019 Blindness Detection
* EfficientNet: Rethinking Model Scaling for CNNs

## Acknowledgments
* APTOS for providing the dataset
* Google Research for EfficientNet architecture
* Kaggle community for insights and discussions

## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/name`)
5. Open Pull Request

## License
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT).
