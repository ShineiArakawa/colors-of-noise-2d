# colors-of-noise-2d

![Teaser](/assets/teaser.png)

A lightweight tool for visualizing 2D colored noise patterns.

## Features

- 2D visualization of different colors of noise
- Interactive adjustment of the power spectrum slope
- Radial power spectrum plotting for frequency analysis

## Requirements

- Python 3.11 or later
- (Optional) CUDA Toolkit 12.0 or later
  - Used for GPU acceleration. If a compatible GPU is unavailable, the code will fall back to CPU execution with OpenMP, although with significantly reduced performance.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/ShineiArakawa/colors-of-noise-2d.git
cd colors-of-noise-2d

pip install -r requirements.txt

# Alternatively, if you use 'uv' for dependency management:
# uv sync
```

## Usage

Launch the visualizer with:

```bash
cd colors-of-noise-2d

python colors_of_noise.py
```

> The first run may take up to ~30 seconds as it compiles the C++/CUDA extensions. Subsequent runs will be much faster thanks to caching.

## License

This project is licensed under the MIT License. See the [LICENSE](/LICENSE.txt) file for details.
