# Credit Analytics Platform 💳

A comprehensive machine learning web application for credit approval prediction and data analytics, built with Streamlit and Python.

## 📋 Overview

This platform provides two main functionalities:
1. **Credit Analytics Dashboard** - Interactive data exploration and visualization
2. **Credit Approval Prediction** - ML-powered loan approval predictions using multiple models

## 🚀 Features

### 📊 Dashboard
- **Statistical Analysis**
  - Comprehensive numeric variable statistics
  - Categorical variable distribution analysis
  - Missing value tracking
  
- **Interactive Visualizations**
  - Histograms with statistical metrics
  - Boxplots for outlier detection
  - Correlation matrix heatmap
  - Scatter plots with custom coloring

### 🔮 Credit Prediction
- **Multi-Model Prediction System**
  - Support Vector Classifier (SVC)
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  
- **User-Friendly Form**
  - Personal information inputs
  - Financial history tracking
  - Loan details specification
  
- **Smart Preprocessing**
  - Automatic feature engineering
  - One-hot encoding for categorical variables
  - Derived financial ratios (debt-to-income, loan-to-income)
  - Age, employment, and credit history binning

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download the project**
   ```bash
   cd C:\Users\Salah\Desktop\ProjetML
   ```

2. **Create a virtual environment (recommended)**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install required packages**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Verify project structure**
   ```
   ProjetML/
   ├── app.py                          # Main application file
   ├── prediction_page.py              # Standalone prediction page
   ├── app3.py                         # Standalone dashboard page
   ├── models.py                       # Custom ML model classes
   ├── Loan_approval_data_2025.csv     # Dataset
   ├── models/                         # Trained model files
   │   ├── pipeline_svc.pkl
   │   ├── pipeline_logistic.pkl
   │   └── pipeline_knn.pkl
   ├── requirements.txt
   └── README.md
   ```

## 🎯 Usage

### Running the Complete Platform

```powershell
streamlit run app.py
```

This launches the integrated application with both Dashboard and Prediction pages accessible via sidebar navigation.

### Running Individual Pages

**Dashboard Only:**
```powershell
streamlit run app3.py
```

**Prediction Only:**
```powershell
streamlit run prediction_page.py
```

### Accessing the Application

Once launched, the application will automatically open in your default web browser at:
```
http://localhost:8501
```

## 📊 Using the Dashboard

1. Navigate to **"📊 Dashboard"** from the sidebar
2. Choose between:
   - **Statistics**: View descriptive statistics and variable distributions
   - **Visualizations**: Explore interactive charts and graphs
3. Select variables from dropdown menus to analyze
4. Interact with plots (zoom, pan, hover for details)

## 🔮 Making Credit Predictions

1. Navigate to **"🔮 Credit Prediction"** from the sidebar
2. Fill in the form with applicant information:
   - **Personal Information**: Age, income, credit score, savings
   - **Financial History**: Employment history, debt, defaults
   - **Loan Details**: Amount, interest rate, purpose
3. Click **"Get Prediction"**
4. View predictions from all three models
5. Download results as CSV if needed

## 🛠️ Technical Details

### Custom ML Models

The project uses custom implementations of ML algorithms defined in `models.py`:
- `LinearSVCScratch`: Custom Support Vector Classifier
- `LogisticRegressionScratch`: Custom Logistic Regression
- `KNN_Scratch`: Custom K-Nearest Neighbors

### Feature Engineering

The preprocessing pipeline automatically generates 35 features from 15 input fields:
- **Derived ratios**: debt-to-income, loan-to-income
- **Age bins**: 26-35, 36-45, 46-55, 56-65, 65+
- **Employment bins**: 2-3, 4-5, 6-10, 11-20, 20+ years
- **Credit history bins**: 2-3, 4-5, 6-10, 11-20, 20+ years
- **One-hot encoding**: occupation status, product type, loan intent

### Model Loading

Models are loaded using `joblib` for efficient serialization. The application gracefully handles missing model files and displays appropriate error messages.

## 📁 Data Format

The application expects a CSV file named `Loan_approval_data_2025.csv` with loan application data containing numeric and categorical variables.

## 🎨 Customization

### Styling
The application uses custom CSS defined in `app.py` for consistent branding:
- Primary color: `#1f77b4` (blue)
- Secondary color: `#ff7f0e` (orange)
- Accent color: `#2ca02c` (green)

### Adding New Models
1. Train your model and save as `.pkl` file in `models/` directory
2. Add model loading logic in the prediction section
3. Update the model display columns accordingly

## 🐛 Troubleshooting

### Common Issues

**Issue**: Model loading errors
- **Solution**: Ensure all `.pkl` files are in the `models/` directory and `models.py` contains the required custom classes

**Issue**: Data file not found
- **Solution**: Verify `Loan_approval_data_2025.csv` is in the project root directory

**Issue**: Import errors
- **Solution**: Ensure all packages are installed: `pip install -r requirements.txt`

**Issue**: Port already in use
- **Solution**: Run with different port: `streamlit run app.py --server.port 8502`

## 📝 Requirements

See `requirements.txt` for complete list of dependencies:
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualizations
- **scikit-learn**: Machine learning utilities
- **joblib**: Model serialization

## 🤝 Contributing

To contribute to this project:
1. Test your changes thoroughly
2. Ensure code follows existing style conventions
3. Update documentation as needed
4. Verify all models load correctly

## 📄 License

This project is created for educational and demonstration purposes.

## 👥 Authors

- **Credit Analytics Platform Team**
- Built with Streamlit v2.0

## 🔗 Links

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

**Version**: 2.0  
**Last Updated**: December 2025  
**Status**: ✅ Production Ready
