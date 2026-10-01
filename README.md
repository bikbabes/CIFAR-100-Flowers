# CIFAR-100 Deep Learning Project: Flowers Class Group

This repository contains our group's deep learning project focusing on the **FLOWERS** coarse class from the CIFAR-100 dataset.

## 🌸 Assigned Dataset Details
- **Coarse Class Label:** Flowers (Label ID: 2)
- **Fine Classes Included:**
  1. Orchid (Label ID: 54 &rarr; Mapped to `0`)
  2. Poppy (Label ID: 62 &rarr; Mapped to `1`)
  3. Rose (Label ID: 70 &rarr; Mapped to `2`)
  4. Sunflower (Label ID: 82 &rarr; Mapped to `3`)
  5. Tulip (Label ID: 92 &rarr; Mapped to `4`)
- **Dataset Scale:** 2,500 Training images | 500 Testing images
- **Image Resolution:** 32x32 pixels, 3 color channels (RGB)

## 📁 Repository Structure
- `extract_flowers.py`: Implements the extraction strategy required by the professor to filter down the dataset and re-map the labels.
- `train_model.py`: Design, compilation, training, and testing logic for our Convolutional Neural Network (CNN).
- `.gitignore`: Configured to keep local environment tracking clean by excluding `.venv`.

## 📈 Current Project Baseline
Our initial basic CNN architecture achieves a baseline performance evaluation:
- **Baseline Test Accuracy:** 58.20% (Trained across 3 initial test epochs)
