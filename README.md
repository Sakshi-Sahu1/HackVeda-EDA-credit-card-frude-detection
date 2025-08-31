# HackVeda-EDA-credit-card-frude-detection
Exploratory Data Analysis on Credit Card Fraud Detection Dataset - Comprehensive EDA with statistical analysis and visualizations


# Credit Card Fraud Detection - Exploratory Data Analysis

## 📋 Project Overview

This project performs comprehensive exploratory data analysis (EDA) on the Credit Card Fraud Detection dataset from Kaggle. The analysis aims to understand data patterns, identify key features for fraud detection, and provide insights for building effective machine learning models.

## 🎯 Assignment Details

- **Assigned On:** 11 Aug 2025 10:26:13
- **Due Date:** 18 Aug 2025
- **Objective:** Perform EDA to gain insights into credit card fraud patterns and prepare data for modeling

## 📊 Dataset Information

- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions with 31 features
- **Features:** 
  - 28 PCA-transformed features (V1-V28)
  - Time: Seconds elapsed between transactions
  - Amount: Transaction amount
  - Class: Target variable (0=Normal, 1=Fraud)

## 🔍 Key Findings

### Class Distribution
- **Normal Transactions:** 284,315 (99.83%)
- **Fraudulent Transactions:** 492 (0.17%)
- **Imbalance Ratio:** 1:577 (highly imbalanced dataset)

### Feature Insights
- **Most Discriminative Features:** V14, V4, V11, V2, V19, V21
- **Amount Analysis:** Fraudulent transactions tend to have lower amounts
- **Time Patterns:** Fraud detection varies throughout the day
- **Data Quality:** No missing values, no duplicates

### Statistical Significance
- All top features show statistically significant differences between fraud and normal transactions
- Strong correlation patterns identified for feature selection

## 🛠️ Technologies Used

- **Python 3.8+**
- **Libraries:**
  - pandas, numpy (Data manipulation)
  - matplotlib, seaborn, plotly (Visualization)
  - scikit-learn (ML utilities)
  - scipy (Statistical tests)

## 📁 Project Structure

```
credit-card-fraud-eda/
│
├── creditcard_fraud_eda.ipynb    # Main EDA Jupyter Notebook
├── eda_report.md                 # Detailed EDA Report
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── setup_instructions.md         # Setup and installation guide
├── data/
│   └── creditcard.csv           # Dataset (download separately)
├── outputs/
│   ├── eda_summary.csv          # Key statistics summary
│   ├── feature_correlations.csv # Feature correlation results
│   └── visualizations/          # Generated plots
└── docs/
    └── methodology.md           # EDA methodology explanation
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/credit-card-fraud-eda.git
cd credit-card-fraud-eda
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
1. Go to [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in the `data/` directory

### 4. Run Analysis
```bash
jupyter notebook creditcard_fraud_eda.ipynb
```

## 📈 Analysis Components

### 1. Data Quality Assessment
- Missing value analysis
- Duplicate detection
- Data type validation
- Basic statistical summary

### 2. Target Variable Analysis
- Class distribution visualization
- Imbalance ratio calculation
- Fraud pattern identification

### 3. Feature Analysis
- Individual feature distributions
- Feature correlation with target
- Statistical significance testing
- Outlier detection using IQR method

### 4. Advanced Visualizations
- Distribution comparisons by class
- Correlation heatmaps
- Scatter plot matrices
- Time series analysis

### 5. Anomaly Detection
- Isolation Forest implementation
- Local Outlier Factor analysis
- Anomaly score evaluation

## 🔬 Methodology

The EDA follows a systematic approach:

1. **Data Loading & Inspection** - Initial data overview
2. **Quality Assessment** - Missing values, duplicates, data types
3. **Univariate Analysis** - Individual feature distributions
4. **Bivariate Analysis** - Feature relationships and correlations
5. **Multivariate Analysis** - Complex pattern identification
6. **Outlier Detection** - Statistical and ML-based methods
7. **Statistical Testing** - Hypothesis testing for significance
8. **Business Insights** - Actionable recommendations

## 📊 Key Visualizations

The notebook generates comprehensive visualizations including:
- Class distribution charts
- Amount distribution comparisons
- Time pattern analysis
- Feature correlation heatmaps
- Outlier detection plots
- Statistical distribution comparisons

## 🎯 Business Recommendations

### For Model Development:
1. **Address class imbalance** using SMOTE or cost-sensitive learning
2. **Focus on top features:** V14, V4, V11, V2, V19 for initial modeling
3. **Use appropriate metrics:** Precision, Recall, F1-score, AUC-ROC
4. **Consider ensemble methods** due to feature complexity

### For Production:
1. **Monitor concept drift** in transaction patterns
2. **Implement real-time anomaly scoring**
3. **Balance false positive costs** with fraud detection accuracy
4. **Regular model retraining** due to evolving fraud patterns

## 📝 Report Summary

The complete EDA reveals:
- **High-quality dataset** with no missing values or duplicates
- **Severe class imbalance** requiring specialized handling techniques
- **Strong discriminative features** available for model training
- **Clear statistical differences** between fraud and normal transactions
- **Multiple outlier patterns** suggesting various fraud types

## 🤝 Contributing

This is an academic project, but improvements are welcome:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📜 License

This project is for educational purposes. The dataset license follows Kaggle terms.

## 📞 Contact: 8959218091

For questions about this analysis, please refer to the detailed methodology in the notebook and report files.

---

**Note:** This EDA was completed as part of an internship assignment focusing on descriptive analytics and data visualization techniques for fraud detection systems.
