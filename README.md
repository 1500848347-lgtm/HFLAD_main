
## 📖 Introduction

In consumer-centric systems such as smart homes, IoT devices, and mobile services, massive amounts of multivariate time series data (e.g., system logs and sensor telemetry) are continuously generated. Accurately and proactively detecting anomalies within this data is crucial for ensuring system security and operational reliability.

Existing methods often process temporal dynamics or feature correlations in isolation, and they struggle to capture patterns across different time scales in long sequences. To address these limitations, we propose **HFLAD (Hierarchical Fusion Learning Anomaly Detection)**, a novel framework focusing on **long-sequence anomaly detection via time and multi-dimensional feature fusion**.

**The core architecture consists of three key modules:**
1. **Hierarchical Time Encoder**: Utilizes causal convolution, dilated convolution, and temporal convolutional networks (TCN) to simultaneously capture macroscopic trends and short-term local fluctuations in long sequences.
2. **Feature Encoder**: Introduces a State-Space Recurrent Neural Network (SRNN) to effectively extract complex interdependencies among multi-dimensional features.
3. **HVAE-Based Generator**: Reconstructs input data through a Hierarchical Variational Autoencoder (HVAE) and utilizes reconstruction errors to precisely quantify and identify anomalies.

## 📁 Directory Structure

The repository is organized as follows for easy reproduction and further development:

```text
HFLAD_main/
├── data/               # Directory for raw target datasets (e.g., SWaT, MSL, KDD-Cup99, ASD)
├── data_processed/     # Scripts for data preprocessing and the processed data
├── models/             # Core HFLAD model definitions (Time Encoder, Feature Encoder, HVAE)
├── main_and_evaluate/  # Scripts for model evaluation and testing
├── pth/                # Directory to save trained model weights (.pth)
├── results/            # Directory to save output results and execution logs
├── utils/              # General utility functions for data loading, metric calculation, etc.
├── train.py            # Main entry script for model training
├── requirements.txt    # List of project environment dependencies
└── *.png               # Experimental result charts from the paper (e.g., Fig6-8, Tables)

```
## 🚀 Quick Start
### 1. Environment Setup
Please ensure your development environment has Python 3.8+ installed. Using a virtual environment is highly recommended:
```
# Clone the repository 
git clone [https://github.com/1500848347-lgtm/HFLAD_main.git](https://github.com/1500848347-lgtm/HFLAD_main.git)
cd HFLAD_main

# Install dependencies
pip install -r requirements.txt
```
### 2. Data Preparation
Create a data/ folder in the root directory and place the downloaded target datasets (e.g., SWaT, MSL) inside. (We recommend downloading the required open-source datasets via cloud drives like [Google Drive](https://drive.google.com/file/d/1nyIRzmkHyqcUlltj2JwCAuWw-h9V9O37/view?usp=sharing)).

Scripts related to data cleaning and preprocessing are located in the data_processed/ directory. Run the corresponding scripts to perform feature engineering and format conversion on the raw data in the data/ directory.

### 3. Model Training
Once the data is ready, run train.py to start training the model. The weight files generated during the training process will be automatically saved in the pth/ folder.
```
python train.py
```
### 4. Model Evaluation
After training, you can execute the run_eval script located in the main_and_evaluate/ directory to calculate core metrics such as Precision, Recall, F1 Score, and AUC on the test set. The evaluation results will be saved in the results/ folder.
```
python main_and_evaluate/run_eval.py
```
