# Setup Instructions for Credit Card Fraud Detection EDA

## 🔧 Environment Setup

### Prerequisites
- Python 3.8 or higher
- Git (for cloning repository)
- Kaggle account (for dataset download)

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
# Clone the project repository
git clone https://github.com/yourusername/credit-card-fraud-eda.git
cd credit-card-fraud-eda
```

#### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv fraud_eda_env

# Activate virtual environment
# On Windows:
fraud_eda_env\Scripts\activate
# On macOS/Linux:
source fraud_eda_env/bin/activate
```

#### 3. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Alternative: Install packages individually
pip install pandas numpy matplotlib seaborn scipy scikit-learn plotly jupyter notebook
```

#### 4. Download Dataset

**Option A: Direct Kaggle Download**
1. Go to [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. Click "Download" (requires Kaggle account)
3. Extract `creditcard.csv` from the zip file
4. Place `creditcard.csv` in the `data/` directory

**Option B: Kaggle API (Advanced)**
```bash
# Install Kaggle API
pip install kaggle

# Configure API credentials (place kaggle.json in ~/.kaggle/)
# Download dataset
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip
mv creditcard.csv data/
```

#### 5. Verify Setup
```bash
# Check if dataset is properly placed
ls data/creditcard.csv

# Start Jupyter Notebook
jupyter notebook

# Open creditcard_fraud_eda.ipynb in the browser
```

## 📁 Project Directory Structure

After setup, your directory should look like:

```
credit-card-fraud-eda/
│
├── creditcard_fraud_eda.ipynb    # Main analysis notebook
├── eda_report.md                 # Detailed report
├── requirements.txt              # Dependencies
├── README.md                     # Project overview
├── setup_instructions.md         # This file
├── methodology.md               # Analysis methodology
│
├── data/
│   └── creditcard.csv           # Dataset (284MB)
│
├── outputs/                     # Generated during analysis
│   ├── eda_summary.csv
│   ├── feature_correlations.csv
│   └── visualizations/
│       ├── class_distribution.png
│       ├── amount_analysis.png
│       ├── correlation_heatmap.png
│       └── time_patterns.png
│
└── docs/
    └── methodology.md
```

## 🚀 Running the Analysis

### 1. Start Jupyter Notebook
```bash
# Navigate to project directory
cd credit-card-fraud-eda

# Start Jupyter
jupyter notebook
```

### 2. Execute Analysis
1. Open `creditcard_fraud_eda.ipynb`
2. Run all cells sequentially (Cell → Run All)
3. Analysis will take 5-10 minutes depending on system

### 3. View Results
- Generated plots will display inline
- Summary files saved to `outputs/` directory
- Final report available in `eda_report.md`

## 🛠️ Troubleshooting

### Common Issues

#### Issue 1: Dataset Not Found
```
FileNotFoundError: creditcard.csv not found
```
**Solution:** Ensure `creditcard.csv` is in the `data/` directory

#### Issue 2: Memory Issues
```
MemoryError: Unable to allocate array
```
**Solution:** 
- Close other applications
- Use data sampling: `df = df.sample(n=50000)` for testing
- Increase virtual memory

#### Issue 3: Package Import Errors
```
ModuleNotFoundError: No module named 'seaborn'
```
**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Issue 4: Jupyter Kernel Issues
```
Kernel appears to have died
```
**Solution:**
- Restart kernel: Kernel → Restart
- Reinstall ipykernel: `pip install --upgrade ipykernel`

### Performance Optimization

#### For Large Dataset Processing:
```python
# Add to notebook if experiencing performance issues
import warnings
warnings.filterwarnings('ignore')

# Optimize pandas
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

# Use sampling for visualization
sample_size = min(10000, len(df))
df_sample = df.sample(n=sample_size, random_state=42)
```

## 📊 Expected Outputs

### Generated Files:
1. **eda_summary.csv** - Key statistics and metrics
2. **feature_correlations.csv** - Feature correlation matrix
3. **Multiple visualizations** - Saved as PNG files

### Analysis Results:
- Comprehensive data quality assessment
- Class imbalance quantification
- Feature importance ranking
- Statistical significance tests
- Outlier detection results
- Business recommendations

## 🔍 Validation Checklist

Before submitting, ensure:

- [ ] Dataset loaded successfully (284,807 rows)
- [ ] All visualizations generated without errors
- [ ] No missing values reported
- [ ] Class imbalance identified (~0.17% fraud)
- [ ] Top correlated features identified
- [ ] Statistical tests completed
- [ ] Outlier analysis performed
- [ ] Summary files exported
- [ ] Report conclusions align with analysis

## 📚 Additional Resources

### Documentation:
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

### Dataset Information:
- [Original Research Paper](https://www.researchgate.net/publication/356799845)
- [Kaggle Dataset Discussion](https://www.kaggle.com/mlg-ulb/creditcardfraud/discussion)

### Further Reading:
- Fraud Detection Techniques
- Handling Imbalanced Datasets
- PCA Feature Interpretation

## 💡 Tips for Success

1. **Run incrementally:** Execute notebook sections step by step
2. **Save frequently:** Use Ctrl+S to save progress
3. **Monitor memory:** Close unnecessary applications
4. **Document insights:** Add markdown cells with observations
5. **Validate results:** Cross-check key findings manually

---

**Setup completed successfully!** You're ready to perform comprehensive EDA on the credit card fraud dataset.
