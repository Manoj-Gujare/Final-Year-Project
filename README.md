# PrescripTech: Generic Prescription Conversion

## Overview

PrescripTech is a healthcare-focused application designed to simplify the conversion of brand-name prescriptions to their generic alternatives. By leveraging advanced deep learning models and OCR technologies, the application accurately detects medication details from prescription images and provides users with cost-effective generic alternatives.

## Features

- **Medicine Detection**: Utilizes the YOLO (You Only Look Once) model to accurately detect columns of medicines in prescription images.
- **Text Extraction**: Implements Tesseract OCR to extract medication details from detected regions.
- **Generic Medicine Conversion**: Matches extracted medications to a comprehensive database of generic equivalents.
- **User-Friendly Interface**: The application is built using Streamlit, providing a simple and interactive web interface for users.
- **Data Validation**: Includes robust data validation checks to ensure the accuracy of extracted medication details, supporting better healthcare decisions.

## Project Structure

- **`.devcontainer/`**: Contains development environment configuration files.
- **`Generic Medicine.csv`**: CSV file with mappings between brand-name medicines and their generic alternatives.
- **`New Model.ipynb`**: Jupyter notebook for the development and experimentation of new models.
- **`PrescripTech_Model.py`**: Main script containing the model logic for prescription analysis.
- **`Procfile.txt`**: Configuration file for deploying the application on Heroku or similar platforms.
- **`Web Scrapping - Generic_Medicine.ipynb`**: Notebook used for scraping generic medicine data from web sources.
- **`app.py`**: The main Python script to run the Streamlit web application.
- **`best.onnx`**: ONNX model file used for efficient inference.
- **`data.yaml`**: Configuration file for YOLO model training parameters.
- **`packages.txt`**: List of required system packages for the project.
- **`requirements.txt`**: List of Python dependencies required for the project.
- **`setup.sh`**: Shell script to set up the environment.
- **`tesseract.exe`**: Executable for Tesseract OCR, used for text extraction.
- **`yolo_training.ipynb`**: Jupyter notebook for training the YOLO model.
