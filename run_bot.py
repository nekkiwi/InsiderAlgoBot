# In run_bot.py (in the project root directory)

import argparse
from src.scraper.feature_scraper import FeatureScraper
from src.scraper.feature_preprocess import FeaturePreprocessor
from src.inference.model_inference import ModelInference
from src.alpaca.alpaca_trader import AlpacaTrader

def main(args):
    """
    Main function to run the complete trading bot pipeline.
    """
    
    ###################
    # Initializations #
    ###################
    
    feature_scraper         = FeatureScraper()
    alpaca_trader           = AlpacaTrader()
    feature_preprocessor    = FeaturePreprocessor()
    model_inference         = ModelInference()
    
    ####################
    # Get Current Data #
    ####################

    print("\n--- Scraping and Preprocessing New Data ---")
    current_features_df = feature_scraper.run(num_business_days=20*12*5)
    if current_features_df is None or current_features_df.empty:
        print("No new data scraped. Exiting.")
        return

    current_features_df_preprocessed = feature_preprocessor.run(current_features_df, args.timepoint, args.threshold_pct)
    if current_features_df_preprocessed is None or current_features_df_preprocessed.empty:
        print("No data available after preprocessing. Exiting.")
        return
        
    #################
    # Run Inference #
    #################
    
    print(f"\n--- Running Inference for Timepoint: {args.timepoint}, Threshold: {args.threshold_pct}% ---")
    results_df = model_inference.run(current_features_df_preprocessed, args.timepoint, args.threshold_pct)

    if results_df is None or results_df.empty:
        print("Inference did not produce results. Exiting.")
        return
        
    ##################
    # Execute Trade #
    ##################
    
    print("\n--- Executing Trades based on Inference Results ---")
    trade_config = {
        "allocation_pct": args.allocation_pct,
        "timepoint": args.timepoint,
        "threshold_pct": args.threshold_pct
    }
    
    print(f"Trade Execution Config: {trade_config}")
    alpaca_trader.run(trade_config, results_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trading bot with specific inference and trading parameters.")
    parser.add_argument("--timepoint", type=str, required=True, help="The prediction timepoint (e.g., '1w', '1m').")
    parser.add_argument("--threshold_pct", type=int, required=True, help="The threshold percentage (e.g., 5 for 5%%).")
    parser.add_argument("--allocation_pct", type=float, required=True, help="The percentage of total portfolio equity to allocate to each trade (e.g., 2.0 for 2%%).")
    args = parser.parse_args()
    main(args)
