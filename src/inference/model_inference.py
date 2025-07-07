import os
import joblib
import pandas as pd
from ast import literal_eval
from sklearn.base import is_classifier
from datetime import timedelta
import time

class ModelInference:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.feature_selection_file = os.path.join(self.data_dir, "analysis/feature_selection/selected_features.xlsx")
        self.models = {}
        self.out_path = ""
        self.selected_features = []
        self.data = {}

    def load_models(self, model, target):
        # TODO: LIMSTOP IS NOT SUPPORTED
        """
        Load 5-fold models for the given target from model_dir.
        Expects files named '{target}_fold{i}.pkl' for i in 1..5.
        """
        model_dir = os.path.join(self.data_dir, f"models/{model}_{target}")
        for i in range(1, 6):
            path = os.path.join(model_dir, f"fold_{i}.joblib")
            self.models[f'fold_{i}'] = joblib.load(path)
        print(f"Loaded models for target '{target}': {list(self.models.keys())}")

    def load_feature_selection(self, target):
        """
        Read selected_features.xlsx, sheet named after target, cell C2 contains comma-separated features.
        """
        df = pd.read_excel(self.feature_selection_file, sheet_name=target)
        raw = df.iloc[0, 2]
        self.selected_features = list(literal_eval(raw))
        print(f"Selected features for '{target}': {self.selected_features}")

    def load_scraped_data(self):
        """
        Load the latest scraped data (e.g. top 20 insider buys) from Excel.
        """
        self.data = pd.read_excel(self.scraped_data_file)
        print(f"Loaded scraped data: {self.data.shape[0]} rows, {self.data.shape[1]} cols")

    def filter_features(self):
        """
        Subset self.data to only identifiers + selected features,
        aligning exactly to training.
        """
        id_cols = ['Ticker', 'Filing Date']
        # 1) move identifiers into index so reset_index won't re-introduce "index"
        df = self.data.set_index(id_cols)
        # 2) reindex to *only* selected_features, filling any missing ones with 0
        df = df.reindex(columns=self.selected_features, fill_value=0)
        # 3) bring identifiers back as columns
        self.data = df.reset_index()
        print(f"Filtered data to {len(self.selected_features)} features + identifiers: {self.data.shape}")



    def run_inference(self, target) -> pd.DataFrame:
        """
        Run inference using the loaded models on filtered data.
        Supports both classifiers (using predict_proba) and regressors (using predict).
        """
        X = self.data[self.selected_features]

        preds = []
        for name, model in self.models.items():
            if is_classifier(model):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            preds.append(pred)

        avg_pred = sum(preds) / len(preds)

        result = self.data[['Ticker', 'Filing Date']].copy()
        result[f'{target}_score'] = avg_pred
        return result

    def run(self, features_df, models, targets):
        """
        Full pipeline: load models, features, data, filter, infer, save.
        """
        start_time = time.time()
        print("\n### START ### Model Inference")
        self.data = features_df
            
        for model in models:
            for target in targets: 
                self.load_models(model, target)
                self.load_feature_selection(target)
                self.filter_features()
                result_df = self.run_inference(target)
        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Model Inference - time elapsed: {elapsed_time}")
        return result_df
