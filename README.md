# Summit Seeker

<p align="center">
  <img src="https://blog.kazaden.com/wp-content/uploads/2016/01/Cerro_torre_1987_compressor-e1452258137301.jpg" alt="Markdownify" width="200">
  <br>
  <a>
    <img src="https://img.shields.io/github/repo-size/aarensberg/summit-seeker">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://badgen.net/pypi/license/pip">
  </a>
</p>

## Overview

Summit Seeker is a machine learning project that helps climbers determine the optimal path on a climbing wall from a photo input. This proof-of-concept (POC) demonstrates the feasibility of using computer vision and path-finding algorithms to assist climbers in planning their routes.

## Features

- **Image Recognition**: Identifies holds on a climbing wall from uploaded photos
- **Path Optimization**: Determines the most efficient climbing route based on difficulty preferences
- **Visualization**: Displays the suggested path overlaid on the original image

## Project Status

This project is currently in the proof-of-concept stage. The core functionality is being developed and tested with a limited dataset of climbing wall images.

## Installation

Clone the repository :
```bash
git clone https://github.com/aarensberg/summit-seeker.git
cd summit-seeker
```

Install dependencies :
```bash
pip install -r requirements.txt
```

## Usage

Coming soon! The application is still in early development.

## Roadmap

- [x] Initial repository setup
- [x] Data cleaning
- [ ] Image preprocessing pipeline
- [ ] Hold detection model
- [ ] Path-finding algorithm
- [ ] API endpoints
- [ ] User interface

## Data sources

- [어니러ㅣ너일 Computer Vision Project](https://universe.roboflow.com/cliving/-axr8m)
- [cliving Computer Vision Project](https://universe.roboflow.com/kmw/cliving)
- [holds Computer Vision Project](https://universe.roboflow.com/mmm-jzxx1/holds-tptrk)
- [HoldSeg Computer Vision Project](https://universe.roboflow.com/ak2isa-lhgcw/holdseg)

All datasets are publicly available. We have merged the data sources listed above into a single file to facilitate model training. You can access to the merged dataset from :

- [Original dataset](https://drive.google.com/drive/folders/1zKj4hAUgME-o-Q5RxZwkgL6TYyriHZmO?usp=sharing) : Contains the four datasets merged together
- [Cleaned dataset](https://drive.google.com/drive/folders/1YGloMA9P_dI6gWHyFkJ2CocDKF1B4ihU?usp=sharing) : Cleaned version of the original dataset (removes duplicates, empty labels, etc.)

## Current Development Focus

### Problem Statement

Our current goal is to train a model capable of recognizing mountain climbing holds in images. However, while there is an abundance of data available for indoor climbing holds (bouldering), there is a significant lack of labeled datasets for outdoor climbing holds (mountain). So far, we have only identified one labeled dataset containing 65 images for outdoor climbing.

### Proposed Solutions

To address this challenge, we are exploring the following approaches:

1. **Transfer Learning with Data Augmentation**:  
  We are initially training the model on the more abundant indoor climbing hold images and then fine-tuning it on the limited outdoor dataset. To enhance the outdoor dataset, we are applying data augmentation techniques such as rotation, scaling, and color adjustments.

2. **Manual Labeling**:  
  If the first approach does not yield satisfactory results, we plan to manually label additional outdoor climbing hold images sourced from the Internet to expand the dataset.

This phase is critical to ensure the model's ability to generalize well to outdoor climbing scenarios.

## Project Structure

```plaintext
summit-seeker/
├── data/
│   └── "boulder/" or "mountain/"
│       └── "test/" or "train/" or "valid"/
|           ├── images/
|           |   └── *.jpg
|           └── labels/
|               └── *.txt     --> YOLO format : <hold_class> <coord_1> <coord_2> <...> <coord_n>
├── notebooks/
│   ├── data_augmentation.ipynb
│   ├── data_cleaning.ipynb
|   └── hold_detector.ipynb
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

The `data/` directory is shown in the repository structure for illustration purposes but is not included in the repository due to size constraints. You can download the datasets directly the Google Drive links provided above.

## Technologies Used

- Python 3.11.9

## License
This project is licensed under the MIT License - see the LICENSE file for details.