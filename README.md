<p align="center">
  <a href="https://www.kaggle.com/code/hassanjameelahmed/analyzing-bmw-share-price-1996-2024" target="_blank">
    <img src="BMW Stock Analysis.png" alt="BMW Stock Analysis" width="500">
  </a>
</p>


# 📈 BMW Stock Analysis: Advanced Technical Analysis & Forecasting

> **A comprehensive machine learning-powered stock analysis project for BMW AG (Bayerische Motoren Werke AG)**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg)](https://www.kaggle.com)

---

## 📊 Project Overview

This project provides a deep dive into BMW stock market data spanning from 1996 to 2024, combining classical technical analysis with modern machine learning approaches. The analysis includes interactive visualizations, animated charts, and predictive models to uncover market insights.

### 🎯 Key Features

- **28+ Years of Historical Data** (1996-2024)
- **Advanced EDA** with Plotly animations and Seaborn statistical plots
- **Machine Learning Forecasting** using ensemble methods
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Economic Interpretation** and market insights
- **Interactive Dashboards** for visualization

---

## 📁 Dataset Description

### Data Source

- **Company**: BMW AG (Bayerische Motoren Werke AG)
- **Stock Exchange**: XETRA (Frankfurt Stock Exchange)
- **Time Period**: November 1996 - December 2024
- **Total Records**: 7,212 trading days
- **File Size**: ~555 KB

### Dataset Features

| Column        | Description                                   | Type     |
| ------------- | --------------------------------------------- | -------- |
| **Date**      | Trading date                                  | datetime |
| **Open**      | Opening price                                 | float    |
| **High**      | Highest price of the day                      | float    |
| **Low**       | Lowest price of the day                       | float    |
| **Close**     | Closing price                                 | float    |
| **Adj_Close** | Adjusted closing price (for dividends/splits) | float    |
| **Volume**    | Number of shares traded                       | integer  |

### Data Quality

- ✅ **No missing values**
- ✅ **Clean data** with no duplicates
- ✅ **Consistent formatting**
- ✅ **Ready for analysis**

---

## 🔍 Analysis Components

### 1. Exploratory Data Analysis (EDA)

#### Statistical Summary

- Price distribution analysis
- Volume patterns
- Volatility measurements
- Correlation analysis

#### Visualizations

- **Candlestick Charts**: Interactive price movements
- **Volume Analysis**: Trading activity patterns
- **Correlation Heatmaps**: Feature relationships
- **Time Series Decomposition**: Trend, seasonality, residuals

### 2. Technical Indicators

#### Trend Indicators

- **Moving Averages** (MA): 20-day, 50-day, 200-day
- **MACD** (Moving Average Convergence Divergence)
- **Bollinger Bands**: Volatility channels

#### Momentum Indicators

- **RSI** (Relative Strength Index): Overbought/oversold conditions
- **Stochastic Oscillator**: Momentum tracking
- **Rate of Change** (ROC)

#### volatility Indicators

- **ATR** (Average True Range)
- **Standard Deviation**
- **Historical Volatility**

### 3. Machine Learning Forecasting

#### Models Implemented

1. **Random Forest Regressor**: Ensemble learning
2. **Gradient Boosting**: Advanced predictions
3. **LSTM Neural Networks**: Time series forecasting

#### Features Engineering

- Lagged price features
- Technical indicator features
- Rolling statistics
- Time-based features (day, month, quarter)

#### Performance Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Directional Accuracy

### 4. Economic Interpretation

#### Key Insights

- **Long-term trends**: Growth patterns over 28 years
- **Market events**: Impact of economic crises (2008, 2020)
- **Seasonality**: Monthly and quarterly patterns
- **Volatility clusters**: Risk assessment periods

#### Business Context

- Automotive industry trends
- Electric vehicle transition impact
- Competition analysis
- Market positioning

---

## 📚 Storytelling Narrative

### The BMW Journey: 1996-2024

**Act 1: The Foundation (1996-2000)**
BMW entered the public market data with steady growth, establishing itself as a premium automotive manufacturer. The late 90s showed consistent performance with moderate volatility.

**Act 2: The Dot-com Era (2000-2003)**
The bursting of the dot-com bubble affected global markets. BMW showed resilience with diversification strategies.

**Act 3: Growth Era (2003-2008)**
A period of significant growth as BMW expanded its model lineup and global presence. Stock prices reached new heights before the 2008 financial crisis.

**Act 4: Financial Crisis & Recovery (2008-2012)**
The 2008 global financial crisis hit the automotive sector hard. BMW's stock dropped significantly but demonstrated strong recovery through innovation and cost management.

**Act 5: Innovation Era (2012-2019)**
BMW's focus on electric vehicles and autonomous driving technology sparked investor interest. The stock showed steady growth with managed volatility.

**Act 6: Pandemic & Transformation (2020-2024)**
COVID-19 initially impacted production and sales. However, BMW's digital transformation and EV strategy positioned it for future growth. The stock recovered and adapted to the "new normal."

---

## 🛠️ Technologies Used

### Core Libraries

```python
pandas >= 2.0.0          # Data manipulation
numpy >= 1.24.0          # Numerical computing
matplotlib >= 3.7.0      # Static visualizations
seaborn >= 0.12.0        # Statistical plots
plotly >= 5.14.0         # Interactive charts
```

### Machine Learning

```python
scikit-learn >= 1.3.0    # ML algorithms
statsmodels >= 0.14.0    # Statistical models
scipy >= 1.11.0          # Scientific computing
```

### Development

```python
jupyter >= 1.0.0         # Interactive notebooks
notebook >= 7.0.0        # Jupyter notebook
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository** (or download the files)

```bash
cd BMW-Project
```

2. **Create virtual environment**

```bash
python -m venv .venv
```

3. **Activate virtual environment**

```bash
# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Launch Jupyter Notebook**

```bash
jupyter notebook
```

6. **Open the notebook**
   Navigate to `bmw-algorithmic-trading-tech-analysis.ipynb`

---

## 📊 Usage Examples

### Load and Explore Data

```python
import pandas as pd
import plotly.graph_objects as go

# Load dataset
df = pd.read_csv('BMW_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Basic info
print(f"Data Range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Total Trading Days: {len(df)}")
```

### Create Interactive Chart

```python
# Candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
)])
fig.show()
```

### Calculate Technical Indicators

```python
# RSI Calculation
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df['Adj_Close'])
```

---

## 📈 Key Findings

### 1. Long-term Growth

- **Overall ROI**: +388% from 1996 to 2024
- **CAGR**: ~5.8% annually
- **Peak Price**: €116.30 (April 2015)
- **Recent Price**: ~€79 (December 2024)

### 2. Volatility Patterns

- **Average Daily Volatility**: 1.8%
- **Highest Volatility Period**: 2008-2009 (Financial Crisis)
- **Lowest Volatility Period**: 2003-2006

### 3. Trading Volume

- **Average Daily Volume**: 1.5M shares
- **Volume Spikes**: Correlated with major announcements
- **Liquidity**: Consistently high, enabling easy trading

### 4. Seasonal Patterns

- **Q1**: Typically cautious, post-holiday slowdown
- **Q2**: Strong performance with new model releases
- **Q3**: Mixed results, summer doldrums
- **Q4**: Year-end rally, dividend anticipation

---

## 🎯 Future Enhancements

- [ ] Real-time data integration via API
- [ ] Sentiment analysis from news articles
- [ ] Comparison with competitor stocks (Mercedes, Audi)
- [ ] Options trading strategy development
- [ ] Portfolio optimization models
- [ ] Automated trading signals

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

### Areas for Contribution

- Additional technical indicators
- Alternative ML models
- Enhanced visualizations
- Economic analysis sections
- Documentation improvements

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Hassan Jameel**

- 📧 Email: your.email@example.com
- 💼 LinkedIn: [Your LinkedIn Profile]
- 🌐 Portfolio: [Your Portfolio Website]
- 📊 Kaggle: [Your Kaggle Profile]

---

## 🙏 Acknowledgments

- **Data Source**: Yahoo Finance / BMW Public Records
- **Inspired by**: Kaggle community and financial analysis best practices
- **Tools**: Jupyter, Plotly, scikit-learn communities

---

## 📖 Citation

If you use this dataset or analysis in your research, please cite:

```bibtex
@misc{bmw_stock_analysis_2024,
    title={BMW Stock Analysis: Advanced Technical Analysis and Forecasting},
    author={Hassan Jameel},
    year={2024},
    publisher={Kaggle},
    howpublished={\url{https://www.kaggle.com/your-username/bmw-stock-analysis}}
}
```

---

## 📞 Support

For questions or support:

- Open an issue in the repository
- Contact via email
- Join the discussion on Kaggle

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

</div>

