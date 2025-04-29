# Summit Seeker

<p align="center">
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

## Technologies Used

- Python 3.11.9

## Usage
Coming soon! The application is still in early development.

## Data sources

- [어니러ㅣ너일 Computer Vision Project](https://universe.roboflow.com/cliving/-axr8m)
- [cliving Computer Vision Project](https://universe.roboflow.com/kmw/cliving)
- [holds Computer Vision Project](https://universe.roboflow.com/mmm-jzxx1/holds-tptrk)
- [HoldSeg Computer Vision Project](https://universe.roboflow.com/ak2isa-lhgcw/holdseg)

All datasets are publicly available [here](https://drive.google.com/drive/folders/1zKj4hAUgME-o-Q5RxZwkgL6TYyriHZmO?usp=sharing). We have merged the data sources listed above into a single file to facilitate model training.

## Roadmap
- [x] Initial repository setup
- [x] Data cleaning
- [ ] Image preprocessing pipeline
- [ ] Hold detection model
- [ ] Path-finding algorithm
- [ ] API endpoints
- [ ] User interface

## Repo Structure
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
|   └── hold_detector.ipynb
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

The `data/` directory is shown in the repository structure for illustration purposes but is not included in the repository due to size constraints. You can download the datasets directly from our [Google Drive](https://drive.google.com/drive/folders/1zKj4hAUgME-o-Q5RxZwkgL6TYyriHZmO?usp=sharing).

## License
This project is licensed under the MIT License - see the LICENSE file for details.