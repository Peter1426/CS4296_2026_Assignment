# CS4296 - MNIST Handwritten Digit Classification with PySpark

## Overview

This is an individual assignment from **CS4296 Cloud Computing** at year 2026. This assignment perform handwritten digit recognition (0-9) using **PySpark MLlib** on the MNIST dataset. Three classification algorithms are implemented and compared:

| Algorithm | Test Accuracy | Training Time |
|-----------|--------------|---------------|
| Logistic Regression | **92.30%** | 15.09s |
| Random Forest | **94.63%** | 22.87s |
| Naive Bayes | **83.65%** | 1.99s |

## Quick Start

### Prerequisites
- Python 3.8+
- Java 8 or 11
- 4GB RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/Peter1426/CS4296_2026_Assignment.git
cd CS4296_2026_Assignment

# Install Python dependencies
pip install -r requirements.txt
```

### Run Locally

The MNIST data is already included in compressed format (.gz) in the data/ folder.

```bash
# Basic run
python src/CS4296_Assignment3_ML.py data/mnist_train.csv.gz data/mnist_test.csv.gz

# Run with output file
python src/CS4296_Assignment3_ML.py data/mnist_train.csv.gz data/mnist_test.csv.gz results.txt
```

### Run on AWS EMR

1. Upload `src/CS4296_Assignment3_ML.py` to S3
2. Upload CSV files to S3
3. Create EMR cluster (`emr-6.15.0`, client mode)
4. Add step:
   - Deploy mode: `Client mode`
   - Application location: `s3://your-bucket/src/CS4296_Assignment3_ML.py`
   - Arguments: `s3://your-bucket/data/mnist_train.csv.gz s3://your-bucket/data/mnist_test.csv.gz`

## Results

### Algorithm Comparison

| Algorithm | Train Accuracy | Test Accuracy | Time (s) |
|-----------|---------------|---------------|----------|
| Logistic Regression | 92.61% | **92.30%** | 15.09 |
| Random Forest | 96.00% | **94.63%** | 22.87 |
| Naive Bayes | 82.53% | **83.65%** | 1.99 |

### Parameter Impact (Logistic Regression)

| Parameter | Value | Train Acc | Test Acc | Time (s) |
|-----------|-------|-----------|----------|----------|
| maxIter | 10 | 91.92% | 91.81% | 4.56 |
| maxIter | 30 | 92.58% | 92.27% | 7.45 |
| maxIter | 50 | 92.61% | **92.30%** | 9.33 |
| maxIter | 100 | 92.64% | 92.29% | 15.23 |
| regParam | 0.000 | 94.03% | 92.44% | 8.89 |
| regParam | 0.001 | 93.74% | **92.64%** | 8.83 |
| regParam | 0.010 | 92.61% | 92.30% | 8.88 |
| regParam | 0.100 | 90.42% | 90.77% | 8.80 |
| regParam | 1.000 | 86.15% | 86.88% | 7.77 |

**Key Findings:**
- `maxIter=50` provides best balance of accuracy and training time
- `regParam=0.001` yields best test accuracy (small regularization)
- Too much regularization (`regParam=1.0`) causes underfitting

## Project Structure

```
CS4296_2026_Assignment/
├── README.md 
├── requirements.txt                         
├── CS4296_Assignment3_Report.pdf  # Assignment report
├── src/
│   └── CS4296_Assignment3_ML.py        # Main classification script
└── data/
    ├── MNIST_test_result_on_EMR.txt    # Test results on AWS EMR                      
    ├── mnist_train.csv.gz              # Training data (compressed, 60,000 samples)
    └── mnist_test.csv.gz               # Test data (compressed, 10,000 samples)
```
### Data structure:

Each CSV file contains:
- **First row:** Header (`label`, `pixel0`, `pixel1`, ..., `pixel783`)
- **Each subsequent row:** One sample (label + 784 pixel values 0-255)
- **Training:** 60,001 rows (header + 60,000 samples)
- **Test:** 10,001 rows (header + 10,000 samples)
