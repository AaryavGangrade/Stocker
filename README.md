# Stocker

A web app for interactive stock price prediction and visualization using linear regression, built with Streamlit.

## Features
- Predicts next-day closing price for any stock ticker (default: AAPL)
- Visualizes price history, moving averages, and model predictions
- Compares model performance to a naive baseline
- Interactive: change ticker and date range in the sidebar

## Getting Started

### 1. Clone the repository
```sh
git clone https://github.com/your-username/stocker.git
cd stocker
```

### 2. Install dependencies
```sh
pip install -r requirements.txt
```

### 3. Run the app
```sh
streamlit run app.py
```

The app will open in your browser. Use the sidebar to select different stocks and date ranges.

## Project Structure
- `app.py` — Main Streamlit app
- `notebooks/` — Jupyter notebook with full data science workflow
- `models/`, `data/`, `figures/` — Supporting files and outputs

## Requirements
- Python 3.8+
- See `requirements.txt` for Python packages

## License
MIT
