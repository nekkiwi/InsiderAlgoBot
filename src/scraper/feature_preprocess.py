import os
import pandas as pd
from datetime import timedelta
import time

from .utils.feature_preprocess_helpers import *

class FeaturePreprocessor:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.data = pd.DataFrame()
        self.continuous_features = None
        self.categorical_features = None
        self.corr_matrix = None
        self.ticker_filing_dates = None
        self.normalization_params = {}
        self.normalization_path = os.path.join(data_dir,'analysis/feature_preprocess/normalization_params.xlsx')

    def prepare_data(self):
        self.data['Filing Date'] = pd.to_datetime(self.data['Filing Date'], dayfirst=True, errors='coerce')
        self.ticker_filing_dates = get_ticker_filing_dates(self.data)
        if self.data is not None and self.ticker_filing_dates is not None:
            self.data.drop(columns=['Ticker', 'Filing Date'], inplace=True)
        else:
            raise ValueError("- Failed to load feature data. Please check the file and its contents.")

    def prepare_filing_dates(self):
        features_cleaned = prepare_filing_dates(self.data, self.ticker_filing_dates)
        return features_cleaned

    def identify_feature_types(self):
        self.categorical_features, self.continuous_features = identify_feature_types(self.data)

        
    def normalize_before_inference(self):
        # Read normalization parameters from Excel
        norm_params = pd.read_excel(self.normalization_path)

        # Ensure proper column naming
        norm_params.columns = ['key', 'min', 'max']
        
        # Clean numerical formatting (handle commas)
        norm_params['min'] = norm_params['min'].replace(",", "")
        norm_params['max'] = norm_params['max'].replace(",", "")

        # Create a dictionary for fast lookup
        normalization_dict = norm_params.set_index('key')[['min', 'max']].to_dict('index')

        # Normalize each column in df based on the params
        for col in self.data.columns:
            if col in normalization_dict:
                min_val = normalization_dict[col]['min']
                max_val = normalization_dict[col]['max']
                # Min-max normalization
                self.data[col] = (self.data[col] - min_val) / (max_val - min_val)


    def run(self, features_df):
        start_time = time.time()
        print("\n### START ### Feature Preprocessing")

        """Run the full analysis pipeline."""
        
        self.data = features_df
            
        self.prepare_data()
        self.identify_feature_types()
        self.normalize_before_inference()
        features_cleaned = self.prepare_filing_dates()

        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Feature Preprocess - time elapsed: {elapsed_time}")

        return features_cleaned

if __name__ == "__main__":
    preprocessor = FeaturePreprocessor()
    preprocessor.run()
