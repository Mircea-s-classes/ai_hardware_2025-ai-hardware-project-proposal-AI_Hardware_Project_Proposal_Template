# AI Hardware Project - The AI Hardware Team
ECE 4332 / ECE 6332 — AI Hardware  
Fall 2025
Grayson Turner
Sammie Levine
Nate Owen
Marissa Cash

## Overview

The goal of this project is to use an Arduino Nano 33 BLE Sense Lite paired with an OV7675 camera module to perform basic facial recognition. The system is designed to detect and identify unique human faces and compare them against a dataset of reference profiles. Based on this face detected the device dhould be able to classfiy each face as either a known resident or an unknown intruder. This simulates the integration of AI into the core logic of a smart doorbell camera system. Our camera system captures images and processses them on the microcontroller and assigns identity labels to different visitors. The design demonstrates how edge solutions of AI hardware can be used in real life scenarios. The final system integrates image acquisition, feature processing and classification to show a complete demonstration of  real time visitor identification. 

Steps to build an image classification system that can identify a known resident or an unknown intruder: 
 1. Collect raw face images using the Arduino Nano 33 BLE Sense Lite and OV7675 camera
 2. Augment the dataset using Google Colab to increase training diversity
 3. Train a Convolutional Neural Network (CNN) using Edge Impulse
 4. Deploy the model for live, on-device facial recognition

## Team Setup
All team members worked together however our official team roles to make sure work was evenly distributed were:
  * **Grayson Turner:** Team Lead - coordination, documentation
  * **Nate Owen:** Hardware - setup, integration
  * **Sammie Levine:** Software - model training, inference 
  * **Marissa Cash:** Evaluation - testing, benchmark

## Implementation

### Software and Hardware Set Up

### Data Collection 

### Run the program 

## Results
The Arduino camera was successfully able to identify between our known resident (Nate), unknown residents, and background data. 

(Accuracy Statisc)

(Result Images)


## Folder Structure
- `docs/` – project proposal and documentation  
- `presentations/` – midterm and final presentation slides  
- `report/` – final written report (IEEE LaTeX and DOCX versions included)  
- `src/` – source code for software, hardware, and experiments  
- `data/` – datasets or pointers to data used


## 📋 Required Deliverables
1. **Project Proposal** — due Nov. 5, 2025, 11:59 PM  
2. **Midterm Presentation** — Nov. 19,2025, 11:59 PM  
3. **Final Presentation and Report** — Dec. 17, 11:59 PM


