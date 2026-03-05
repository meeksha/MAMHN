# Memory-Augmented Meta-Hypernetwork for Adaptive Anomaly Detection on Edge Devices

## Overview

Edge and IoT devices continuously generate telemetry data from sensors and network systems. Detecting anomalies in such environments is challenging because device behavior and network traffic patterns change over time. Traditional anomaly detection models often fail when the underlying data distribution shifts.

This project implements an adaptive anomaly detection system that combines a **hypernetwork**, a **memory module**, and **meta-learning** to dynamically adapt anomaly detection models to changing IoT data patterns.

The system processes IoT telemetry data, retrieves past patterns from memory, and dynamically generates model parameters to improve anomaly detection performance.

---

## Dataset

This project uses **IoT telemetry datasets inspired by the TON_IoT framework**, which simulate sensor data from IoT environments such as weather monitoring systems and device telemetry.

The dataset includes multiple numerical features representing IoT sensor behavior. Artificial spikes and abnormal patterns are introduced to simulate anomaly conditions.

Example dataset files used during experimentation:

* IoT_Fake_16features.csv
* IoT_Fake_16features_Spiked.csv
* IoT_Garage_Spiked.csv
* IoT_Garage_Spiked_Renamed.csv
* IoT_Weather_Fake.csv

During preprocessing, the dataset is converted into training, validation, and testing arrays.

Feature scaling is performed using **MinMaxScaler**, which is stored as:

* minmax_scaler.joblib

---

## Methodology

The anomaly detection framework is composed of three main components.

### Hypernetwork

A hypernetwork is a neural network that generates the weights of another neural network. Instead of using a static anomaly detection model, the hypernetwork dynamically produces model parameters based on the current data distribution.

### Memory Module

The memory module stores representations of previously observed data patterns. When new input data is processed, the system retrieves similar patterns from memory to improve prediction.

### Meta-Learning

Meta-learning allows the system to learn how to adapt quickly to new environments with minimal retraining.

### System Pipeline

IoT Telemetry Data
→ Data Preprocessing and Scaling
→ Memory Module
→ Meta-Hypernetwork
→ Adaptive Anomaly Detection Model
→ Prediction (Normal or Anomaly)

---

## Project Structure

app.py
Main application used to run the anomaly detection system.

experiment_summary.json
Contains training results and experiment metrics.

retrain_summary.json
Contains retraining experiment results.

.gitignore
Specifies files and directories ignored by Git.

README.md
Project documentation.

---

## Installation

Clone the repository:

git clone https://github.com/meeksha/MAMHN.git

Move into the project directory:

cd MAMHN

Install required Python libraries:

pip install numpy pandas torch scikit-learn joblib flask

---

## Running the Project

Run the anomaly detection system using:

python app.py

The system will load the trained models, preprocess IoT telemetry data, generate adaptive detector parameters using the hypernetwork, and classify anomalies.

---

## Applications

This approach can be applied in several real-world scenarios:

* IoT security monitoring
* industrial sensor anomaly detection
* smart home monitoring systems
* edge computing environments
* infrastructure monitoring systems

---

## Future Work

Potential improvements for the system include:

* real-time streaming anomaly detection
* deployment on edge hardware devices
* continual learning for long-term adaptation
* integration with distributed IoT monitoring systems

---

## Author

A S Sameeksha
B.E Artificial Intelligence and Data Science
Siddaganga Institute of Technology
