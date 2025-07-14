# In run_bot.py (in the project root directory)

import argparse
import pandas as pd
from src.scraper.feature_scraper import FeatureScraper
from src.scraper.feature_preprocess import FeaturePreprocessor
from src.inference.model_inference import ModelInference
from src.alpaca.alpaca_trader import AlpacaTrader

def get_holding_period(timepoint: str) -> int:
    """Converts a timepoint string (e.g., '1w', '1m') to a holding period in days."""
    if 'w' in timepoint:
        return int(timepoint.replace('w', '')) * 7
    if 'm' in timepoint:
        return int(timepoint.replace('m', '')) * 30
    return 30

def main(args):
    """
    Main function to run the complete trading bot pipeline.
    """
    WINNING_MODEL_TYPE = "LightGBM"
    WINNING_CATEGORY = "alpha"
    
    ###################
    # Initializations #
    ###################
    
    print("--- Initializing Bot Components ---")
    feature_scraper = FeatureScraper()
    alpaca_trader   = AlpacaTrader()

    # --- FIX: Initialize FeaturePreprocessor with strategy parameters ---
    feature_preprocessor = FeaturePreprocessor()

    # --- FIX: Simplified ModelInference initialization ---
    model_inference = ModelInference(
        model_type=WINNING_MODEL_TYPE,
        category=WINNING_CATEGORY,
        timepoint=args.timepoint,
        threshold_pct=args.threshold_pct
    )
    
    ####################
    # Get Current Data #
    ####################

    print("\n--- Scraping and Preprocessing New Data ---")
    current_features_df = feature_scraper.run(num_days=1)
    if current_features_df is None or current_features_df.empty:
        print("No new data scraped. Exiting.")
        return

    # --- FIX: Call preprocessor in inference mode ---
    current_features_df_preprocessed = feature_preprocessor.run(current_features_df, args.timepoint, args.threshold_pct)
    if current_features_df_preprocessed is None or current_features_df_preprocessed.empty:
        print("No data available after preprocessing. Exiting.")
        return
        
    #################
    # Run Inference #
    #################
    
    print(f"\n--- Running Inference for Timepoint: {args.timepoint}, Threshold: {args.threshold_pct}% ---")
    results_df = model_inference.run(inference_df=current_features_df_preprocessed)

    if results_df is None or results_df.empty:
        print("Inference did not produce results. Exiting.")
        return
        
    ##################
    # Execute Trade #
    ##################
    
    print("\n--- Executing Trades based on Inference Results ---")
    trade_config = {
        "amount": 100,
        "holding_period": get_holding_period(args.timepoint),
        "timepoint": args.timepoint,
        "threshold": args.threshold_pct
    }
    
    print(f"Trade Execution Config: {trade_config}")
    alpaca_trader.run(trade_config, results_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trading bot with specific inference and trading parameters.")
    parser.add_argument("--timepoint", type=str, required=True, help="The prediction timepoint (e.g., '1w', '1m').")
    parser.add_argument("--threshold_pct", type=int, required=True, help="The threshold percentage (e.g., 2 for 2%%).")
    args = parser.parse_args()
    main(args)
