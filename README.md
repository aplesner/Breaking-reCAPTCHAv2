# Breaking-reCAPTCHAv2
Code for the paper `Breaking reCAPTCHAv2` accepted at COMPSAC 2024 (https://arxiv.org/abs/2409.08831).

## Introduction
CAPTCHAs (Completely Automated Public Turing Tests to Tell Computers and Humans Apart) have been a vital security measure on the internet, protecting websites from automated bots and malicious activities. However, with the rapid advancements in machine learning and artificial intelligence, the effectiveness of CAPTCHAs in distinguishing between humans and machines has come into question.

This semester project focuses on Google's reCAPTCHAv2 system, which is widely used across the web. We aim to analyze the effectiveness of reCAPTCHAv2 in rejecting bots using advanced deep learning models such as YOLO (You Only Look Once). Our research explores the vulnerabilities of image-based CAPTCHAs and develops efficient methods to solve them using state-of-the-art machine learning techniques.

Our main findings include:

- We can solve 100% of the CAPTCHAs presented by reCAPTCHAv2, surpassing the success rates of previous works, which range from 68% to 71%.
- There is no significant difference in the number of challenges required by humans and bots to solve CAPTCHAs in reCAPTCHAv2, suggesting that current AI technologies can effectively exploit advanced image-based CAPTCHAs.
- reCAPTCHAv2 heavily relies on cookie and browser history data when evaluating whether a user is human or not, highlighting the importance of considering user-specific data in CAPTCHA design.


| CAPTCHA Example | CAPTCHA Result |
|:---:|:---:|
| <img src="assets/type2_example.gif" width="200" /> | <img src="assets/type2_example_result.png" width="200" /> |

## Requirements
- Python 3.9
- Firefox (Geckodriver)
- Required libraries (see `requirements.txt`)

## Installation
1. Clone the repository

2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

3. Download and set up Geckodriver:
   - Download Geckodriver from the official website: [Geckodriver Releases](https://github.com/mozilla/geckodriver/releases)
   - Extract the downloaded archive and add the path to the `geckodriver` executable to your system's PATH environment variable.

## Usage
- Run `solve_recaptcha.py` to solve a single reCAPTCHA challenge:
  ```
  python solve_recaptcha.py
  ```

- Run `test_environment.py` to solve multiple reCAPTCHA challenges and create a log file:
  ```
  python test_environment.py
  ```

## Data
- The training data for the classification task can be found [here](https://drive.google.com/drive/folders/19kET6PFXHaHZqzr9DU_ZsgX-n13Ef4sj?usp=sharing).
- The validation data for the classification task can be found [here](https://drive.google.com/drive/folders/19kG2NQls2iH1sUq0js0MOArBiJiOmVGM?usp=sharing).

## Project Structure
This project has the following directory structure:

- `assets/`: Contains the images used in the README.
- `IP/`: Contains the script for changing IP address.
  - `vpn.py`: Script for changing IP address (only works on macOS).
- `models/`: Contains all the models used in the project.
  - `YOLO_Classification/`: Contains the YOLO model for classification.
  - `YOLO_Segment/`: Contains the YOLO model for segmentation.
- `utils/`: Contains utility scripts.
  - `collect_data.py`: Script for collecting data for the classification model.
  - `label_tool.py`: A UI tool for labeling CAPTCHA images.
  - `visualize_log_files.py`: Analyzes log files, providing insights into task attempts, their distribution, and key statistical measures.
- `README.md`: This file, providing an overview of the project.
- `solve_recaptcha.py`: Demo script for solving a single reCAPTCHA challenge using Selenium.
- `test_environment.py`: Demo script for solving multiple reCAPTCHA challenges using Selenium and creating a log file.
- `requirements.txt`: Contains the required libraries for the project.

## System Compatibility
Please note that the code in this project was developed and tested on macOS M1. While most of the code should be compatible with other operating systems, certain functionalities, such as the VPN script (`vpn.py`), may not work on non-macOS systems. If you encounter any issues running the code on a different operating system, please refer to the documentation of the specific libraries or tools used in the project for guidance on how to set them up for your system.

