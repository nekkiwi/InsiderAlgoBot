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
    convert_timepoints_to_bdays,
    get_bot_bought_tickers
)

class AlpacaTrader:
    def __init__(self):
        load_dotenv()
        self.client = REST(
            os.getenv("ALPACA_API_KEY"),
            os.getenv("ALPACA_API_SECRET_KEY"),
            "https://paper-api.alpaca.markets",
        )
        self.sheet_name = ""

    def sell_matured(self, holding_business_days: int):
        sell_matured_positions(self.client, holding_business_days, self.sheet_name)

    @staticmethod
    def read_signals(results_df: pd.DataFrame) -> pd.DataFrame:
        buy_signals_df = results_df[results_df['Final_Signal'] == 1].copy()
        if buy_signals_df.empty:
            return pd.DataFrame(columns=["symbol", "score"])
        out = buy_signals_df[['Ticker', 'Predicted_Return']].copy()
        # Sort by prediction score so the best signal is first
        buy_signals_df.sort_values(by='Predicted_Return', ascending=False, inplace=True)
        # Drop duplicate tickers, keeping only the first (highest score) one
        unique_signals_df = buy_signals_df.drop_duplicates(subset='Ticker', keep='first')
        print(f"Found {len(unique_signals_df)} unique 'buy' signals after de-duplication.")
        
        out = unique_signals_df[['Ticker', 'Predicted_Return']].copy()
        out.columns = ["symbol", "score"]
        return out
    
    def buy_new(self, symbols, amount_per_trade: float, results_df=None):
        placed = False
        bot_buy_history = get_bot_bought_tickers(self.sheet_name)
        
        # Also get currently held positions to avoid re-buying something that hasn't been sold yet.
        held_symbols = {p.symbol for p in self.client.list_positions()}
        
        for sym in symbols:
            if sym in bot_buy_history:
                print(f"ℹ️  Skipping {sym}: Already in this bot's historical buys (from '{self.sheet_name}' sheet).")
                continue
                
            if place_order(self.client, sym, amount_per_trade, results_df, self.sheet_name):
                placed = True
        
        return placed

    def run(self, config: dict, results_df: pd.DataFrame = None):
        start = time.time()
        print("\n### START ### Alpaca Trader")

        self.sheet_name = f"{config['timepoint']}-{config['threshold_pct']}%"
        print(f"### Logging to sheet: '{self.sheet_name}' ###")

        holding_business_days = convert_timepoints_to_bdays(config['timepoint'])

        # 1) Sell any matured positions
        self.sell_matured(holding_business_days)

        # 2) Load & filter signals
        signals = self.read_signals(results_df)

        if signals.empty:
            print("No valid 'buy' signals found in inference results.")
            log_to_google_sheet("No new good buy found", self.sheet_name)
        else:
            try:
                account_info = self.client.get_account()
                equity = float(account_info.equity)
                allocation_pct = config["allocation_pct"]
                amount_per_trade = equity * (allocation_pct / 100.0)
                
                print(f"- Portfolio Equity: ${equity:,.2f}")
                print(f"- Allocation per Trade: {allocation_pct:.2f}%")
                print(f"- Calculated Amount per Trade: ${amount_per_trade:,.2f}")

                symbols = signals["symbol"].tolist()
                if not self.buy_new(symbols, amount_per_trade, results_df):
                    log_to_google_sheet("No new good buy found", self.sheet_name)
            
            except Exception as e:
                print(f"ERROR: Could not get account equity or calculate trade amount: {e}")
                log_to_google_sheet(f"Failed to execute buys due to equity calculation error: {e}", self.sheet_name)


        elapsed = timedelta(seconds=int(time.time() - start))
        print(f"### END ### Elapsed: {elapsed}")
