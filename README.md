
# PaySafeGNN - Online Payment Fraud Detection with GNNs and Tabular ML

## Overview
This project implements a robust fraud detection pipeline on a large-scale online payment transactions dataset using both classic tabular machine learning and modern Graph Neural Networks (GNNs). It demonstrates how conventional ML models (Logistic Regression, XGBoost, Random Forest) and graph-based approaches can complement each other to better identify fraudulent patterns in financial data.

## Features
- **End-to-End Data Processing:** Automated handling, cleaning, and encoding of tabular transaction data.[1]
- **Tabular ML Models:** ROC-AUC-optimized training for Logistic Regression, XGBoost, and Random Forest models.[1]
- **Graph Construction & GNN:** Transform transactions and account relationships into a network, applying PyTorch Geometric for node classification.[1]
- **Performance Visualization:** Confusion matrix, classification report, ROC/PR curves, and graph structure visualizations using NetworkX and Matplotlib.[1]
- **Extensible Design:** Clearly documented code for adaptation to related fraud analysis or graph-based financial tasks.[1]

***

## Dataset

| Column            | Description                                | Type       |
|-------------------|--------------------------------------------|------------|
| step              | Time step of transaction                   | Integer    |
| type              | Transaction type (TRANSFER, CASH_IN, etc.) | Category   |
| amount            | Amount transferred                         | Float      |
| nameOrig/nameDest | Pseudonymous user/account IDs              | String     |
| oldbalanceOrg     | Sender's account pre-transaction           | Float      |
| newbalanceOrg     | Sender's account post-transaction          | Float      |
| oldbalanceDest    | Receiver's account pre-transaction         | Float      |
| newbalanceDest    | Receiver's account post-transaction        | Float      |
| isFraud           | 1 = Fraudulent, 0 = Legitimate             | Integer    |
| isFlaggedFraud    | Heuristic flag for fraud                   | Integer    |

- **Size:** 83,561 rows, 11 columns


***

## Workflow

### 1. Data Preprocessing
- Encodes categorical variables, imputation for missing values, and splits dataset into train/test ensuring reproducibility.
- Feature engineering for ML and GNN graph construction with node and edge mapping based on account relationships.[1]

### 2. Tabular ML Model Training
- Trains Logistic Regression, XGBoost, and Random Forest classifiers.
- Calculates ROC-AUC accuracy for benchmarking model performance.
- Model Evaluation: Outputs Validation Accuracy and confusion matrix per classifier.

### 3. Graph Neural Network (GNN) Implementation
- Builds financial transaction graph: nodes are accounts; edges represent transactions.
- Aggregates node features (e.g., transaction counts per account).
- Trains a two-layer GCN (Graph Convolution Network) for account node classification.
- Evaluates GNN with classification metrics and visual output (test accuracy, confusion matrix, ROC/PR curves).

### 4. Visualization
- Plots transaction distribution and correlation heatmaps with seaborn.
- Visualizes constructed graph structure, node labels, and prediction correctness using NetworkX and matplotlib.
- Displays sampled subgraph with node coloring to indicate prediction status.

***

## Example Results

- **Tabular ML Models:**
    - Logistic Regression ROC-AUC: **0.92**
    - XGBoost ROC-AUC: **0.99**
    - Random Forest ROC-AUC: **0.87**

- **GNN Node Classification:**
    - Test Accuracy: **0.99** (with dummy node labels)
    - Extensive feature documentation for future fraud-focused GNN tasks

- **Graph Visualization:**
    - Interactive network diagrams for accounts and transactions
    - Visual mapping of prediction correctness in sampled subgraphs

***

## Usage

```bash
# Install requirements
pip install torch torch_geometric sklearn pandas matplotlib seaborn networkx

# Run the main pipeline (in Jupyter or as script)
python fraud_detection.py
```

Data and scripts are organized for modular adaptation. See code comments and `.ipynb` for detailed step-by-step usage.[1]

***


## Future Improvements

- Integrate domain knowledge to define nuanced node and edge features for GNNs.
- Use transaction-level fraud labels for edge/node tasks, or explore link prediction for risky relationships.
- Extend evaluation to handle class imbalance and deploy the model as a live fraud prevention tool.

***

## Citation

If using this repository or dataset, please cite as:

> Online Payment Fraud Detection with Graph Neural Networks and Tabular ML Classifiers, 2025 (see repo or dataset for details).

***

## Contact

For further collaboration or AI/ML research guidance, please reach out via [GitHub Issues] or direct email.

***

## Acknowledgements

Special thanks to open-source ML/graph libraries and Kaggle contributors for dataset support.[1]

***

## License

MIT License. See LICENSE file for more details.

***

This README presents all core aspects of the project, including practical results, full workflow, and future directions, supporting professional and academic sharing.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/83134720/bbd9e41f-db5c-42de-9d88-10857b185e0b/vertopal.com_OnlinePaymentFraudDetection-1.pdf)
