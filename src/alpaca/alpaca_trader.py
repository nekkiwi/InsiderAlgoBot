# In src/alpaca/alpaca_trader.py

import os
import time
from datetime import timedelta
import pandas as pd
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST
from src.alpaca.utils.alpaca_trader_helpers import (
    log_to_google_sheet,
    sell_matured_positions,
    place_order,
)

class AlpacaTrader:
    def __init__(self):
        load_dotenv()
        self.client = REST(
            os.getenv("ALPACA_API_KEY"),
            os.getenv("ALPACA_API_SECRET_KEY"),
            "https://paper-api.alpaca.markets",
        )

    def sell_matured(self, holding_days: int):
        sell_matured_positions(self.client, holding_days)

    @staticmethod
    def read_signals(results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reads signals from the inference results DataFrame. A signal is valid
        only if the 'Final_Signal' column is 1.
        """
        # --- FIX: Filter directly on the 'Final_Signal' column ---
        buy_signals_df = results_df[results_df['Final_Signal'] == 1].copy()

        if buy_signals_df.empty:
            return pd.DataFrame(columns=["symbol", "score"])

        # Use Ticker for symbol and Predicted_Return as the score for logging
        out = buy_signals_df[['Ticker', 'Predicted_Return']].copy()
        out.columns = ["symbol", "score"]
        return out

    def buy_new(self, symbols, amount: float, results_df=None):
        placed = False
        for sym in symbols:
            if place_order(self.client, sym, amount, results_df):
                placed = True
        return placed

    def run(self, config: dict, results_df: pd.DataFrame = None):
        start = time.time()
        print("\n### START ### Alpaca Trader")

        # 1) Sell any matured positions
        self.sell_matured(config["holding_period"])

        # 2) Load & filter signals
        # --- FIX: Logic is simplified to trust the inference output ---
        signals = self.read_signals(results_df)

        if signals.empty:
            print("No valid 'buy' signals found in inference results.")
            log_to_google_sheet("No new good buy found")
        else:
            symbols = signals["symbol"].tolist()
            if not self.buy_new(symbols, config["amount"], results_df):
                log_to_google_sheet("No new good buy found")

        elapsed = timedelta(seconds=int(time.time() - start))
        print(f"### END ### Elapsed: {elapsed}")
