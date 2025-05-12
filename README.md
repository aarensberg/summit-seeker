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
- [x] Data augmentation
- [x] Hold detection model
- [x] Path-finding algorithm
- [ ] User interface

## Data sources

### Original Datasets

The 4 datasets below were the original database we wanted to use. The problem is that we wanted to use an object detection model to detect holds on outdoor climbing routes. However, these datasets mainly contained images of indoor climbing boulders, which didn't suit our purpose. Nevertheless, we decided to try training a model on these data to increase the size of our dataset, which was insufficient.

1. [어니러ㅣ너일 Computer Vision Project](https://universe.roboflow.com/foad-ad5491-gmail-com/climbing-dataset-ekl0f)
2. [cliving Computer Vision Project](https://universe.roboflow.com/kmw/cliving)
3. [holds Computer Vision Project](https://universe.roboflow.com/mmm-jzxx1/holds-tptrk)
4. [HoldSeg Computer Vision Project](https://universe.roboflow.com/ak2isa-lhgcw/holdseg)

All datasets are publicly available. We have merged the data sources listed above into a single file to facilitate model training. You can access to the merged dataset from :

[Google Drive](https://drive.google.com/drive/folders/1zKj4hAUgME-o-Q5RxZwkgL6TYyriHZmO?usp=sharing) : Contains the four datasets merged together

### Current Dataset

Since the model wasn't learning correctly on the block data, we decided to keep only the “Climbing dataset”, which contains images of outdoor climbing routes but very few of them. We therefore proceeded to increase the size of our dataset from 45 training images to 990. That's 22 augmented images per original image.

1. [Climbing dataset](https://universe.roboflow.com/foad-ad5491-gmail-com/climbing-dataset-ekl0f)

[Google Drive](https://drive.google.com/drive/folders/1YGloMA9P_dI6gWHyFkJ2CocDKF1B4ihU?usp=sharing) : Cleaned version of the original dataset (removes duplicates, empty labels, etc.)

## Project Structure

```plaintext
summit-seeker/
├── data/
│   └── "augmented_train/" or "test/" or "train/" or "valid/"
|       ├── images/
|       |   └── *.jpg
|       └── labels/
|           └── *.txt     --> YOLO format : <hold_class> <coord_1> <coord_2> <...> <coord_n>
├── notebooks/
│   ├── data_augmentation.ipynb
│   ├── data_cleaning.ipynb
│   ├── model_training.ipynb
|   └── path_finder.ipynb
├── scripts/
|   └── model_training.py
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