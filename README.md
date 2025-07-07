# InsiderAlgoBot

**InsiderAlgoBot** automates the process of scraping recent insider trading data, engineering useful features, running machine learning models and executing trades via the Alpaca API. It is organised as a set of independent modules that can be run together as a daily pipeline or individually for experimentation.

## Features

- **Data Scraper** – retrieves insider transaction data from `openinsider.com` and augments it with technical indicators, financial ratios and recent insider activity.
- **Feature Preprocessor** – cleans and normalises the scraped data so that it matches the model training setup.
- **Model Inference** – loads pre‑trained models to generate scores for target metrics such as one‑month return.
- **Alpaca Trader** – places buy and sell orders on Alpaca based on the model results and configurable thresholds.

## Installation

1. Install Python 3.10 or newer.
2. Clone this repository and install the dependencies:

   ```bash
   pip install -r requirements.txt
   pip install alpaca-trade-api==3.2.0 --no-deps
   ```

3. Create a `.env` file in the project root with your Alpaca credentials:

   ```dotenv
   ALPACA_API_KEY=<your_key>
   ALPACA_API_SECRET_KEY=<your_secret>
   ```

The existing GitHub Action (`.github/workflows/insideralgobot.yml`) shows the minimal setup used in CI.

## Usage

Run the full pipeline locally:

```bash
python run_bot.py
```

This executes the following steps:

1. **FeatureScraper** – downloads recent insider trades and computes technical/financial features.
2. **FeaturePreprocessor** – prepares the dataset and applies normalisation.
3. **ModelInference** – loads models from `data/models/` and produces inference results.
4. **AlpacaTrader** – places trades using the generated signals.

Adjust trading parameters in `run_bot.py` or pass a custom config when calling `AlpacaTrader.run()`.

## Repository Structure

```
├── data/                 # Models and analysis artefacts
├── src/
│   ├── scraper/          # Scraping and feature engineering
│   ├── inference/        # Model loading and inference helpers
│   └── alpaca/           # Trading logic for Alpaca API
├── run_bot.py            # Example orchestration script
└── requirements.txt      # Python dependencies
```

## Contributing

Contributions are welcome. Please open an issue or submit a pull request if you have improvements or bug fixes.

## License

This project is provided as-is without a specific license. Please contact the maintainer for details before use in production.
