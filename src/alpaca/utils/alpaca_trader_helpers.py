from finvizfinance.quote import finvizfinance
import gspread
import json
import os
from google.oauth2.service_account import Credentials
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import time
import numpy as np
import re

def convert_timepoints_to_bdays(timepoint) -> dict:
    """
    Converts a list of timepoint strings (e.g., '1w', '2m') into a
    dictionary mapping the timepoint to its equivalent number of business days.
    - 1 week ('w') = 5 business days
    - 1 month ('m') = 20 business days
    """
    converted = 0
    match = re.match(r"(\d+)([dwmy])", timepoint.lower())
    if not match:
        raise ValueError(f"Invalid timepoint format: '{timepoint}'. Use formats like '5d', '1w', '3m'.")
    
    num, unit = int(match.group(1)), match.group(2)
    
    if unit == 'd':
        converted = num
    elif unit == 'w':
        converted = num * 5
    elif unit == 'm':
        converted = num * 20  # Approximate business days in a month
    elif unit == 'y':
        converted = num * 240 # Approximate business days in a year (12 * 20)
    else:
        raise ValueError(f"Unknown time unit: '{unit}' in timepoint '{timepoint}'")
            
    return converted

def calculate_business_days(start_date: datetime, end_date: datetime) -> int:
    """
    Number of business days _between_ start_date and end_date,
    excluding the end date itself. E.g.:

      Mon→Tue  → 1  
      Mon→Wed  → 2  
      Fri→Mon  → 1  (skips weekend)  
    """
    return np.busday_count(start_date.date(), end_date.date())

def get_latest_buy_order(client, symbol):
    return client.list_orders(
        status='closed',
        symbols=[symbol],
        side='buy',
        limit=1,
        direction='desc'
    )

def make_timezone_aware(dt):
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def get_price_and_quantity(client, symbol, amount):
    """
    Returns latest price and quantity of shares to buy for a given amount.
    """
    latest_trade = client.get_latest_trade(symbol)
    price = latest_trade.p
    qty_to_buy = int(amount // price)
    return price, qty_to_buy

def submit_sell_order(client, symbol, qty):
    return client.submit_order(
        symbol=symbol,
        qty=qty,
        side='sell',
        type='market',
        time_in_force='day'
    )

def submit_buy_order(client, symbol, qty):
    return client.submit_order(
        symbol=symbol,
        qty=qty,
        side='buy',
        type='market',
        time_in_force='day'
    )

def get_fundamentals_and_prediction(ticker: str, results_df) -> str:
    """
    Get P/E, P/S, D/E, and prediction score for a given ticker and return formatted string.
    """
    def parse_ratio(val):
        try:
            return round(float(val), 2)
        except (ValueError, TypeError):
            return "NA"

    try:
        stock = finvizfinance(ticker)
        info = stock.ticker_fundament()
        pe, ps, de = parse_ratio(info.get("P/E")), parse_ratio(info.get("P/S")), parse_ratio(info.get("Debt/Eq"))
    except Exception:
        pe, ps, de = "NA", "NA", "NA"

    try:
        # --- FIX: Use the 'Predicted_Return' column from the results_df ---
        pred_score = results_df.loc[results_df['Ticker'] == ticker, 'Predicted_Return'].values[0]
        pred_score = round(float(pred_score), 4)
    except (IndexError, ValueError, TypeError):
        pred_score = "NA"

    return f"P/E={pe}, P/S={ps}, D/E={de}, Pred={pred_score}"

def log_to_google_sheet(message: str, sheet_name: str):
    """Logs a message to a specific worksheet in the Google Sheet."""
    load_dotenv()
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    creds_dict = json.loads(os.getenv("GOOGLE_SHEET_CREDS_JSON"))
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)

    sh = gc.open("InsiderAlgoBot - Log")

    # Get or create the specific worksheet
    try:
        worksheet = sh.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        print(f"Worksheet '{sheet_name}' not found. Creating it...")
        worksheet = sh.add_worksheet(title=sheet_name, rows="1000", cols="20")
        worksheet.append_row(["Date", "Time", "Message"], value_input_option="RAW")

    cet = timezone(timedelta(hours=2))
    now = datetime.now(cet)
    date_str = now.strftime("%d/%m/%Y")
    time_str = now.strftime("%H:%M")

    worksheet.append_row([date_str, time_str, message], value_input_option="RAW")
    
def sell_matured_positions(client, holding_business_days: int, sheet_name: str):
    print("- Checking for positions to sell...")
    try:
        # Get the list of tickers this specific bot is allowed to sell
        bot_owned_tickers = get_bot_bought_tickers(sheet_name)
        if not bot_owned_tickers:
            print("ℹ️  This bot has no buy history. No positions will be sold.")
            return

        positions = client.list_positions()
        if not positions:
            print("ℹ️  No open positions to check.")
            return

        clock = client.get_clock()
        market_open = clock.is_open

        for position in positions:
            # *** CORE LOGIC CHANGE: Only check positions this bot owns ***
            if position.symbol not in bot_owned_tickers:
                print(f"ℹ️  Skipping {position.symbol}: Not owned by bot '{sheet_name}'.")
                continue

            try:
                buy_orders = get_latest_buy_order(client, position.symbol)
                if not buy_orders:
                    print(f"ℹ️  No buy order found for {position.symbol}.")
                    continue
                
                buy_order = buy_orders[0]
                purchase_ts = make_timezone_aware(buy_order.filled_at)
                buy_price = float(buy_order.filled_avg_price) if buy_order.filled_avg_price else None

                held_bdays = calculate_business_days(purchase_ts, datetime.now(timezone.utc)) + 1 # buy order was executed on hold day 1

                if held_bdays < holding_business_days:
                    print(f"✅ {position.symbol} held {held_bdays} business days — within holding period of {holding_business_days} b-days.")
                    continue

                if market_open:
                    print(f"ℹ️  Selling {position.symbol}, held {held_bdays} days.")
                    sell_order = submit_sell_order(client, position.symbol, position.qty)
                    
                    deadline = datetime.now(timezone.utc) + timedelta(minutes=5)
                    sell_price = None
                    while datetime.now(timezone.utc) < deadline:
                        o = client.get_order(sell_order.id)
                        if o.filled_avg_price:
                            sell_price = float(o.filled_avg_price)
                            break
                        time.sleep(5)

                    if sell_price is None:
                        log_to_google_sheet(f"Sell for {position.symbol} did not fill in time.", sheet_name)
                    elif buy_price is not None:
                        ret = (sell_price - buy_price) / buy_price
                        ret_pct = round(ret * 100, 2)
                        log_message = (f"Sold {position.qty} {position.symbol} at ${sell_price:.2f} "
                                       f"(bought at ${buy_price:.2f}). Return: {ret_pct}%")
                        log_to_google_sheet(log_message, sheet_name)
                        print(f"✅ {log_message}")
                    else:
                        log_to_google_sheet(f"Sold {position.symbol} at ${sell_price:.2f}, but could not calculate return.", sheet_name)
                else:
                    print("⚠️  Market is closed. Cannot sell.")
                    log_to_google_sheet("Sell to be executed but market is closed.", sheet_name)
                    break 
            except Exception as e:
                log_to_google_sheet(f"Error processing position {position.symbol}: {e}", sheet_name)

    except Exception as e:
        log_to_google_sheet(f"Error fetching positions: {e}", sheet_name)
        
        
def place_order(client, symbol: str, amount: float, results_df, sheet_name: str):
    print("- Checking for stocks to buy...")
    order_placed = False
    held_symbols = {p.symbol for p in client.list_positions()}
    open_orders = client.list_orders(status='open')
    open_buy_symbols = {o.symbol for o in open_orders if o.side == 'buy'}

    if symbol in held_symbols or symbol in open_buy_symbols:
        return order_placed

    try:
        price, qty_to_buy = get_price_and_quantity(client, symbol, amount)
        if qty_to_buy <= 0:
            return order_placed

        submit_buy_order(client, symbol, qty_to_buy)
        order_placed = True
        
        # Ensure total value is formatted correctly
        total_value = f"{price * qty_to_buy:.2f}"
        log_message = f"Buy executed: {symbol} for ${total_value}"
        if results_df is not None:
            details = get_fundamentals_and_prediction(symbol, results_df)
            log_message = f"{log_message}, {details}"
        log_to_google_sheet(log_message, sheet_name)

    except Exception as e:
        print(f"Error buying {symbol}: {e}")
    return order_placed

def get_bot_bought_tickers(sheet_name: str) -> list[str]:
    """
    Reads the log for a specific bot and returns a list of tickers
    it has bought.
    """
    print(f"Reading buy history from sheet: '{sheet_name}'...")
    load_dotenv()
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    try:
        creds_dict = json.loads(os.getenv("GOOGLE_SHEET_CREDS_JSON"))
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open("InsiderAlgoBot - Log")
        
        worksheet = sh.worksheet(sheet_name)
        records = worksheet.get_all_records() # Gets records as a list of dicts
        
        bought_tickers = set()
        for record in records:
            message = record.get("Message", "")
            if message.startswith("Buy executed:"):
                # Extracts the ticker, e.g., from "Buy executed: AAPL for 150.00$"
                parts = message.split()
                if len(parts) > 2:
                    bought_tickers.add(parts[2])
        
        print(f"Found {len(bought_tickers)} unique tickers bought by this bot.")
        return list(bought_tickers)
        
    except gspread.exceptions.WorksheetNotFound:
        print(f"Worksheet '{sheet_name}' not found. Assuming no buy history.")
        return []
    except Exception as e:
        print(f"An error occurred while reading Google Sheet: {e}")
        return []