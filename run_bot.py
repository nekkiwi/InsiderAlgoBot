from src.scraper.feature_scraper import FeatureScraper
from src.scraper.feature_preprocess import FeaturePreprocessor
from src.inference.model_inference import ModelInference
from src.alpaca.alpaca_trader import AlpacaTrader

def main():
    ###################
    # Initializations #
    ###################
    
    feature_scraper         = FeatureScraper()
    feature_preprocessor    = FeaturePreprocessor()
    model_inference         = ModelInference()
    alpaca_trader           = AlpacaTrader()
    
    current_features_df_preprocessed = None
    
    ####################
    # Get Current Data #
    ####################

    current_features_df = feature_scraper.run(num_weeks=1)
    current_features_df_preprocessed = feature_preprocessor.run(current_features_df)
    
    #################
    # Run Inference #
    #################
    
    models  = ['RandomForestOverSample']
    targets = ['final_return_1m_raw']
    
    results_df = model_inference.run(current_features_df_preprocessed, models, targets)
    
    ##################
    # Excecute Trade #
    ##################
    
    config = {
        "amount": 100,
        "holding_period": 30,
        "target": "final_return_1m_raw",
        "threshold": 0.06,
    }
    
    alpaca_trader.run(config, results_df)
    
if __name__ == "__main__":
    main()