import os
import glob
import joblib
import pandas as pd
import numpy as np
import re

class ModelInference:
    def __init__(self):
        """
        Initializes the inference engine with the core strategy parameters.
        The 'top_n' and 'optimize_for' parameters are no longer needed, as all artifacts
        are loaded from the strategy's dedicated directory.
        """
        self.model_type = "LightGBM"
        self.category = "alpha"
        self.timepoint = ""
        self.threshold_pct = 0
        
        # Define base directories
        self.base_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.final_models_dir = os.path.join(self.base_dir, 'models')
        self.output_dir = os.path.join(self.base_dir, 'inference')

    def _load_final_artifacts(self):
        """
        Loads all final artifacts (models, features, and optimal threshold) 
        from a single, self-contained strategy directory.
        """
        models = {}
        strategy_dir_name = f"{self.model_type}_{self.category}_{self.timepoint}_{self.threshold_pct}pct"
        model_dir = os.path.join(self.final_models_dir, strategy_dir_name)

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Final model directory not found: {model_dir}")

        # --- Load all artifacts from the strategy directory ---
        features_path = os.path.join(model_dir, "final_features.joblib")
        threshold_path = os.path.join(model_dir, "optimal_threshold.joblib")

        if not os.path.exists(features_path) or not os.path.exists(threshold_path):
            raise FileNotFoundError(f"Required artifacts (features or threshold) not found in {model_dir}")

        final_features = joblib.load(features_path)
        optimal_threshold = joblib.load(threshold_path)
        
        print(f"- Loaded {len(final_features)} features and optimal threshold ({optimal_threshold:.4f}) from '{strategy_dir_name}'.")

        # --- Load all model pairs (classifier and regressor) ---
        clf_dir = os.path.join(model_dir, "classifier_weights")
        reg_dir = os.path.join(model_dir, "regressor_weights")
        
        clf_files = glob.glob(os.path.join(clf_dir, "final_clf_seed*.joblib"))
        reg_files = glob.glob(os.path.join(reg_dir, "final_reg_seed*.joblib"))

        if not clf_files:
            raise FileNotFoundError(f"No classifier models found in {clf_dir}")

        for f in clf_files:
            seed = int(re.search(r'seed(\d+)', f).group(1))
            models[seed] = {'clf': joblib.load(f)}
        
        for f in reg_files:
            seed = int(re.search(r'seed(\d+)', f).group(1))
            if seed in models:
                models[seed]['reg'] = joblib.load(f)

        print(f"- Loaded {len(models)} final model pairs.")
        return models, final_features, optimal_threshold

    def run(self, inference_df: pd.DataFrame, timepoint, threshold_pct):
        """
        Full inference pipeline using the self-contained strategy artifacts.
        """
        self.timepoint = timepoint
        self.threshold_pct = threshold_pct
        # --- 1. Load all artifacts from one location ---
        models, final_features, optimal_threshold = self._load_final_artifacts()
        
        if inference_df is None or inference_df.empty:
            print("Inference data is empty. Nothing to predict.")
            return None
            
        # --- 2. Prepare inference data ---
        # The 'FeaturePreprocessor' should have already prepared the data.
        # This step ensures the columns are in the exact same order as during training.
        X_inference = inference_df.reindex(columns=final_features, fill_value=0)

        # --- 3. Run two-stage inference, averaging predictions across all seeds ---
        all_clf_probas = [m['clf'].predict_proba(X_inference)[:, 1] for m in models.values()]
        all_reg_preds = [m['reg'].predict(X_inference) for m in models.values() if 'reg' in m]

        avg_clf_proba = np.mean(all_clf_probas, axis=0)
        avg_reg_pred = np.mean(all_reg_preds, axis=0) if all_reg_preds else np.zeros_like(avg_clf_proba)
        
        # --- 4. Generate final signals and save the output ---
        output_df = inference_df[['Ticker', 'Filing Date']].copy()
        output_df['Classifier_Positive_Probability'] = avg_clf_proba
        output_df['Predicted_Return'] = avg_reg_pred
        
        # A buy signal requires both a positive class and a predicted return above the optimal threshold
        output_df['Final_Signal'] = (
            (output_df['Classifier_Positive_Probability'] > 0.5) & 
            (output_df['Predicted_Return'] >= optimal_threshold)
        ).astype(int)
        
        print(f"\nInference complete. {output_df['Final_Signal'].sum()} 'buy' signals generated.")
        print(f"\nFull Inference results: \n{output_df}")
        # print(f"Results saved to: {output_path}")

        return output_df
