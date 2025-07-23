# ML Analysis Tool - Streamlit Version

A modern, interactive machine learning analysis tool built with Streamlit that provides automated feature engineering, model comparison, and AI-powered insights.

## 📁 Project Structure

```
ml-analysis-tool/
├── app.py                       # Main Streamlit app
├── data_processor.py            # Data loading and preprocessing logic
├── feature_engineer.py          # Feature engineering operations
├── model_comparator.py          # Model comparison and evaluation
├── ml_analyzer.py               # Orchestrates the full ML analysis pipeline
├── llm_recommender.py           # Gemini-powered AI recommendations
├── utils.py                     # Utility functions
├── requirements.txt             # Python dependencies
├── sample_data/                 # Built-in sample datasets (CSV files)
│   ├── Churn_Modelling.csv
│   ├── house_price_regression_dataset.csv
│   ├── used_car_price_data.csv
│   ├── Multiclass Diabetes Dataset.csv
│   ├── parkinsons_disease_dataset.csv
│   └── Student_Performance.csv
└── README.md                    # Project documentation
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment
```bash
cp env_example.txt .env
```
Edit `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Run the App
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 🎯 Features

- **Upload Data**: Drag and drop CSV/Excel files or use Quick Start to load built-in sample datasets. Automatic data quality analysis, sample preview, and missing value detection.
- **Drop Columns**: Select columns to exclude from model training (e.g., IDs, names, irrelevant features).
- **Select Target**: Choose the column to predict (from remaining columns).
- **One-click Analysis**: Real-time progress tracking and comprehensive model comparison.
- **Results Visualization**: Interactive charts and tables for model performance, feature importance, and engineering steps.
- **AI Insights**: Gemini-powered recommendations and business guidance.

## 📊 Supported Models

**Classification:** Logistic Regression, Random Forest, Gradient Boosting, SVM, K-Nearest Neighbors, Decision Tree, Naive Bayes

**Regression:** Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, SVR, K-Nearest Neighbors, Decision Tree

## 🔧 Configuration & Analysis

- Automatic problem detection (classification vs regression)
- **Intelligent Feature Scaling**: Chooses between StandardScaler, MinMaxScaler, and RobustScaler based on outlier analysis (IQR method)
- Feature engineering and selection based on data characteristics
- Comprehensive model evaluation with multiple metrics

## 🧪 Sample Data

Sample datasets are included in the `sample_data/` directory and can be loaded directly from the app's UI.

## 🐛 Troubleshooting

- **Streamlit not starting**: Check dependencies and Python version (3.8+ recommended)
- **Gemini API errors**: Ensure API key is set in `.env`, check internet and quota
- **File upload issues**: Use CSV/Excel, check file size (max 16MB), verify file integrity
- **Analysis failures**: Ensure target column exists, enough data, and no missing values in target
- **Performance tips**: For large datasets (>10,000 rows), consider sampling; use efficient data types; close other apps

## 🔄 Workflow

1. **Upload Dataset** or use Quick Start
2. **Review Data**
3. **Drop Columns**
4. **Select Target**
5. **Run Analysis**
6. **Review Results**
7. **Get AI Insights**
8. **Export Results**

## 🎯 Use Cases

- Data Scientists: Quick model comparison and feature analysis
- Business Analysts: Understanding data patterns and model performance
- Students: Learning ML concepts with real data
- Researchers: Rapid prototyping and model evaluation

## 📝 License

This project is licensed under the MIT License.