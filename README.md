# MO-GEOMETRY: Virtual Camera

## Overview

This project demonstrates the intersection of conic sections and non-linear maps using a virtual camera. The application is built using Python and various libraries such as OpenCV, PIL, and Matplotlib.

## Features

- **Virtual Camera**: Simulates a virtual camera to project and manipulate images.
- **ChArUco Board Detection**: Integrates ChArUco board detection for camera calibration and image manipulation.
- **Interactive GUI**: Provides an interactive GUI for importing images, manipulating them, and visualizing the results.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/mo-geometry/conic_sections.git
    cd conic_sections
    ```

2. **Create a virtual environment**:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the application**:
    ```sh
    python main_app.py
    ```

2. **Import an image**: Use the GUI to import an image for manipulation.

3. **Toggle ChArUco Board**: Enable or disable the ChArUco board detection for calibration.

4. **Manipulate Image**: Use the provided tools to manipulate the image and visualize the results.

## Project Structure

- `main_app.py`: Main application file.
- `app/`: Contains the application modules and resources.
- `modules/`: Contains the core modules for camera, world, photon, etc.
- `images/`: Directory for storing images used in the application.

## References

- B. O'Sullivan and P. Stec, "Sensor tilt via conic sections" IMVIP pages 141-144 (2020).
- P. Stec and B. O'Sullivan, ["Method for compensating for the off axis tilting of a lens"](https://patents.google.com/patent/US10356346) United States patent number: 10,356,346 B1, July (2019).
