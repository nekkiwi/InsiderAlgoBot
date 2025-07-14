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

def save_feature_data(data, ticker_filing_dates, file_path, train):
    """Save the processed feature data."""
    data_dir = os.path.join(os.path.dirname(__file__), '../../../data')
    ticker_filing_dates['Filing Date'] = pd.to_datetime(ticker_filing_dates['Filing Date'], dayfirst=True, errors='coerce')
    ticker_filing_dates.dropna(subset=['Filing Date'], inplace=True)
    ticker_filing_dates['Filing Date'] = ticker_filing_dates['Filing Date'].dt.strftime('%d/%m/%Y %H:%M')
    final_data = pd.concat([ticker_filing_dates, data], axis=1)

    if train:
        file_path = os.path.join(data_dir, file_path)
        if not final_data.empty:
            try:
                final_data.to_excel(file_path, index=False)
                print(f"- Data successfully saved to {file_path}.")
            except Exception as e:
                print(f"- Failed to save data to Excel: {e}")
        else:
            print("- No data to save.")
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

def engineer_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers new, more powerful features from the existing feature set.

    Args:
        df (pd.DataFrame): The input DataFrame from features_final.xlsx.

    Returns:
        pd.DataFrame: The DataFrame with new, engineered features added.
    """
    print("- Engineering new features...")
    
    # Use a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Define a small epsilon to prevent division-by-zero errors
    epsilon = 1e-6

    # --- Strategy 1: Interaction Features (Role x Value) ---
    # These features isolate the transaction value for the most important roles.
    df['CEO_Buy_Value'] = df['Value'] * df['CEO']
    df['CFO_Buy_Value'] = df['Value'] * df['CFO']
    df['Pres_Buy_Value'] = df['Value'] * df['Pres']
    
    # --- Strategy 2: Consolidated Insider Importance Score ---
    # This creates a single powerful feature summarizing the insider's rank.
    df['Insider_Importance_Score'] = (
        3 * df['CEO'] + 
        3 * df['Pres'] + 
        2 * df['CFO'] + 
        1 * df['Dir']
    )

    # --- Strategy 3: Ratio and Momentum Features ---
    # Ratio of recent purchases to sales. High values indicate strong buying pressure.
    # df['Purchase_Sale_Ratio_Quarter'] = df['num_purchases_quarter'] / (df['num_sales_quarter'] + epsilon)

    # Normalizes the transaction value by the company's market cap.
    df['Value_to_MarketCap'] = df['Value'] / (df['Market_Cap'] + epsilon)

    # Captures "buying the dip" vs. "buying at new highs".
    df['Distance_from_52W_High'] = 1 - df['52_Week_High_Normalized']
    
    print(f"- Successfully added {len(['CEO_Buy_Value', 'CFO_Buy_Value', 'Pres_Buy_Value', 'Insider_Importance_Score', 'Purchase_Sale_Ratio_Quarter', 'Value_to_MarketCap', 'Distance_from_52W_High'])} new features.")
    
    return df
