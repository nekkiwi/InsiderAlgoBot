# InsiderAlgoBot

InsiderAlgoBot is an automated trading pipeline that monitors recent insider transactions, scores opportunities with machine-learning models, and optionally executes trades through the Alpaca API. Each stage can be run independently for analysis or combined into a daily workflow.

## Features

- **Data Scraper** – collects insider trading reports from [openinsider.com](https://openinsider.com) and enriches them with technical indicators and basic financial ratios.
- **Feature Preprocessor** – aligns scraped data to the same format used for model training and loads the necessary scalers and feature lists.
- **Model Inference** – applies pre-trained LightGBM models to generate buy signals and expected returns.
- **Alpaca Trader** – places market orders on Alpaca and logs activity to Google Sheets.

## Getting Started

### Requirements

- Python 3.10+
- An Alpaca paper trading account
- Google service-account credentials (for optional logging)

### Installation

```bash
pip install -r requirements.txt
pip install alpaca-trade-api==3.2.0 --no-deps
```

Create a `.env` file in the repository root containing your credentials:

```dotenv
ALPACA_API_KEY=<your_key>
ALPACA_API_SECRET_KEY=<your_secret>
# Optional: required for logging and model download
GOOGLE_SHEET_CREDS_JSON=<json_credentials>
GDRIVE_FOLDER_ID=<google_drive_folder>
```

Models should be placed under `data/models/` using the naming convention\
`LightGBM_<category>_<timepoint>_<threshold_pct>pct` (see the GitHub workflows for examples).

## Running the Bot

Use `run_bot.py` to scrape new data, run inference, and execute trades:

```bash
python run_bot.py --timepoint "1w" --threshold_pct 0
```

The script performs the following steps:

1. **FeatureScraper** – downloads and processes the latest insider trades.
2. **FeaturePreprocessor** – normalises the features using the saved scalers.
3. **ModelInference** – generates predictions from the pre-trained models.
4. **AlpacaTrader** – submits orders and logs results.

Adjust the amount, time horizon, and threshold directly in `run_bot.py` or via command-line arguments.

> **Weights**: This repository does not include model or fold weights. To run the bot or reproduce the evaluation, please reach out to obtain the required weight files.

## GitHub Actions

Two sample workflows (`insideralgobot_1w_0pct.yml` and `insideralgobot_3m_10pct.yml`) demonstrate how to schedule the bot on GitHub. They download models from Google Drive, install dependencies, and run the pipeline daily.

## Project Structure

```
├── data/                 # Pre-trained models and analysis artifacts
├── src/
│   ├── scraper/          # Scraping and feature engineering
│   ├── inference/        # Model loading and inference helpers
│   └── alpaca/           # Trading logic for Alpaca API
├── run_bot.py            # Example orchestration script
└── requirements.txt      # Python dependencies
```

## Contributing

Pull requests and issues are welcome. Please ensure any code changes are tested before submission.

## License

This project is provided as-is without a specific license. Contact the maintainer for details before any production use.

