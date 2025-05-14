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

Lauch image detection model and open path-finding interface :
```bash
python -m app.main <image_path>
```

options :
- `--model` : Specify a different path for the YOLO model (default: [PACTEv3.pt](models/PACTEv3.pt))
- `--show-detections` : Display raw detections before path calculation
- `--conf` : Modify confidence threshold for detections (value between 0 and 1)

## Roadmap

- [x] Initial repository setup
- [x] Data cleaning
- [x] Data augmentation
- [x] Hold detection model
- [x] Path-finding algorithm
- [x] User interface (not perfect yet but functional)

## Data sources

### Original Datasets

The 4 datasets below were the original database we wanted to use. The problem is that we wanted to use an object detection model to detect holds on outdoor climbing routes. However, these datasets mainly contained images of indoor climbing boulders, which didn't suit our purpose. Nevertheless, we decided to try training a model on these data to increase the size of our dataset, which was insufficient.

1. [climbing-dataset Computer Vision Project](https://universe.roboflow.com/foad-ad5491-gmail-com/climbing-dataset-ekl0f)
2. [cliving Computer Vision Project](https://universe.roboflow.com/kmw/cliving)
3. [holds Computer Vision Project](https://universe.roboflow.com/mmm-jzxx1/holds-tptrk)
4. [HoldSeg Computer Vision Project](https://universe.roboflow.com/ak2isa-lhgcw/holdseg)

The dataset contains 8981 (64+810+7996+111) images:
- 8196 (45+717+7335+99) training images
- 292 (13+60+211+8) validation images
- 493 (6+33+450+4) test images

All datasets are publicly available. We have merged the data sources listed above into a single file to facilitate model training. You can access to the merged dataset from :

[Google Drive](https://drive.google.com/drive/folders/1zKj4hAUgME-o-Q5RxZwkgL6TYyriHZmO?usp=sharing) : Contains the four datasets merged together

### Current Dataset

Since the model wasn't learning correctly on the block data, we decided to keep only the “Climbing dataset”, which contains images of outdoor climbing routes but very few of them. We therefore proceeded to increase the size of our dataset from 45 training images to 990. That's 22 augmented images per original image.

1. [climbing-dataset Computer Vision Project](https://universe.roboflow.com/foad-ad5491-gmail-com/climbing-dataset-ekl0f)

The dataset contains 64 images:
- 45 training images
- 13 test images
- 6 validation images

[Google Drive](https://drive.google.com/drive/folders/1YGloMA9P_dI6gWHyFkJ2CocDKF1B4ihU?usp=sharing) : Cleaned version of the original dataset (removes duplicates, empty labels, etc.)

## Project Structure

```plaintext
summit-seeker/
├── app/
│   ├── __init__.py
│   ├── hold_detector.py
│   ├── main.py
|   └── path_finder.py
├── data/
│   └── "augmented_train/" or "test/" or "train/" or "valid/"
|       ├── images/
|       |   └── *.jpg
|       └── labels/
|           └── *.txt     --> YOLO format : <hold_class> <coord_1> <coord_2> <...> <coord_n>
├── models/
│   ├── 2025-05-09_13h46_yolo11n.pt
│   ├── PACTEv1.pt
│   ├── PACTEv2.pt
|   └── PACTEv3.pt
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

## License
This project is licensed under the MIT License - see the LICENSE file for details.