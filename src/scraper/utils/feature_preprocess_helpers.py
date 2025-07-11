import os
import pandas as pd

# Helper functions for loading, saving, and identifying feature types

def load_feature_data(file_path):
    """Load the feature data from an Excel file and extract Ticker and Filing Date."""
    data_dir = os.path.join(os.path.dirname(__file__), '../../../data')
    file_path = os.path.join(data_dir, file_path)
    if os.path.exists(file_path):
        try:
            data = pd.read_excel(file_path)
            print(f"- Sheet successfully loaded from {file_path}.")
            return data
        except Exception as e:
            print(f"- Failed to load sheet from {file_path}: {e}")
            return None
    else:
        print(f"- File '{file_path}' does not exist.")
        return None


def get_ticker_filing_dates(data):
    """Extract Ticker and Filing Date."""
    ticker_filing_dates = data[['Ticker', 'Filing Date']].copy()
    ticker_filing_dates['Filing Date'] = ticker_filing_dates['Filing Date'].dt.strftime('%d/%m/%Y %H:%M')
    return ticker_filing_dates

def prepare_filing_dates(data, ticker_filing_dates):
    """Save the processed feature data."""
    ticker_filing_dates['Filing Date'] = pd.to_datetime(ticker_filing_dates['Filing Date'], dayfirst=True, errors='coerce')
    ticker_filing_dates.dropna(subset=['Filing Date'], inplace=True)
    ticker_filing_dates['Filing Date'] = ticker_filing_dates['Filing Date'].dt.strftime('%d/%m/%Y %H:%M')
    final_data = pd.concat([ticker_filing_dates, data], axis=1)
    return final_data

def identify_feature_types(df):
    """
    Very simple split:
      - Categorical: columns whose set of non-null values is exactly {0,1}, 
                     or whose dtype is object/category/bool.
      - Continuous:  all other numeric columns.
      - Everything else: categorical.
    Returns (categorical_cols, continuous_cols).
    """
    categorical_cols = []
    continuous_cols  = []
    
    for col in df.columns:
        ser = df[col]
        # drop nulls for the value‐check
        vals = set(ser.dropna().unique())
        
        # 1) 0/1‐only → categorical
        if vals == {0, 1}:
            categorical_cols.append(col)
        
        else:
            continuous_cols.append(col)
    
    return categorical_cols, continuous_cols