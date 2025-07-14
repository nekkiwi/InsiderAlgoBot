import os
import pandas as pd
from datetime import timedelta
import time
import joblib

# Import helper functions
from .utils.feature_preprocess_helpers import *

class FeaturePreprocessor:
    def __init__(self):
        """
        Initializes the preprocessor. For inference, strategy parameters are required
        to load the correct artifacts.
        """
        self.data = pd.DataFrame()
        self.ticker_filing_dates = None
        self.continuous_features = None
        self.categorical_features = None
        self.corr_matrix = None

        # --- Parameters for loading inference artifacts ---
        self.model_type = "LightGBM"
        self.category = "alpha"

        # --- Base directory for locating model artifacts ---
        self.base_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.models_dir = os.path.join(self.base_dir, 'models')
        
        # --- Placeholders for loaded artifacts ---
        self.final_scaler = None
        self.final_features = None

    def _load_inference_artifacts(self):
        """Loads the final scaler and feature list for a specific model strategy."""
        if not all([self.model_type, self.category, self.timepoint, self.threshold_pct is not None]):
            raise ValueError("For inference, model_type, category, timepoint, and threshold_pct must be provided during initialization.")

        strategy_dir_name = f"{self.model_type}_{self.category}_{self.timepoint}_{self.threshold_pct}pct"
        strategy_path = os.path.join(self.models_dir, strategy_dir_name)

        scaler_path = os.path.join(strategy_path, 'final_scaler.joblib')
        features_path = os.path.join(strategy_path, 'final_features.joblib')

        if not os.path.exists(scaler_path) or not os.path.exists(features_path):
            raise FileNotFoundError(f"Inference artifacts not found in '{strategy_path}'. Ensure the final model has been trained for this strategy.")

        print(f"- Loading inference artifacts from: {strategy_dir_name}")
        self.final_scaler = joblib.load(scaler_path)
        self.final_features = joblib.load(features_path)

    def prepare_data(self):
        self.data['Filing Date'] = pd.to_datetime(self.data['Filing Date'], dayfirst=True, errors='coerce')
        self.ticker_filing_dates = get_ticker_filing_dates(self.data)
        cols_to_drop = [col for col in ['Ticker', 'Filing Date'] if col in self.data.columns]
        if cols_to_drop:
            self.data.drop(columns=cols_to_drop, inplace=True)

    def save_feature_data(self, file_path, train):
        features_cleaned = save_feature_data(self.data, self.ticker_filing_dates, file_path, train)
        return features_cleaned

    def identify_feature_types(self):
        self.categorical_features, self.continuous_features = identify_feature_types(self.data)

    def run(self, features_df: pd.DataFrame, timepoint, threshold_pct):
        start_time = time.time()
        print("\n### START ### Feature Preprocessing")

        self.data = features_df.copy()
        
        self.timepoint = timepoint
        self.threshold_pct = threshold_pct
            
        # Feature engineering is applied to both training and inference data first.
        self.data = engineer_new_features(self.data)
        self.prepare_data()

        # The inference path loads artifacts and applies them without discovery.
        print("- Running preprocessing for INFERENCE...")
        self._load_inference_artifacts()

        # Align columns to match the exact feature set used for training
        print(f"- Aligning data to the {len(self.final_features)} features used for training.")
        self.data = self.data.reindex(columns=self.final_features, fill_value=0)

        # Identify continuous features from the final, aligned feature set
        _, continuous_features_to_scale = identify_feature_types(self.data)
        
        # Apply the saved scaler to the continuous features
        if continuous_features_to_scale:
            print(f"- Applying final scaler to {len(continuous_features_to_scale)} continuous features.")
            self.data[continuous_features_to_scale] = self.final_scaler.transform(self.data[continuous_features_to_scale])
        
        # Re-attach Ticker/Date for output and return the processed DataFrame
        features_cleaned = self.save_feature_data('', train=False)

        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Feature Preprocess - time elapsed: {elapsed_time}")

        return features_cleaned