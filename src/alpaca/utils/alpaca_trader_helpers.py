from finvizfinance.quote import finvizfinance
import gspread
import json
import os
from google.oauth2.service_account import Credentials
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import time

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

def log_to_google_sheet(message: str):
    load_dotenv()
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    creds_dict = json.loads(os.getenv("GOOGLE_SHEET_CREDS_JSON"))
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)

    # Open sheet and worksheet
    sh = gc.open("InsiderAlgoBot - Log")
    worksheet = sh.sheet1
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%d/%m/%Y")   # e.g. “07/07/2025”
    time_str = now.strftime("%H:%M")      # e.g. “14:35”

    # This will write Date | Time | Message
    worksheet.append_row([date_str, time_str, message], value_input_option="RAW")
    
def sell_matured_positions(client, holding_period_days):
    print("- Checking for positions to sell...")
    try:
        positions = client.list_positions()
        if not positions:
            print("ℹ️  No open positions to check.")
            return

        # check market clock once
        clock = client.get_clock()
        market_open = clock.is_open

        for position in positions:
            try:
                # 1) get your original buy
                buy_orders = get_latest_buy_order(client, position.symbol)
                if not buy_orders:
                    print(f"ℹ️  No buy order found for {position.symbol}.")
                    continue
                buy_order   = buy_orders[0]
                purchase_ts = make_timezone_aware(buy_order.filled_at)

                # 2) holding check
                held_days = (datetime.now(timezone.utc) - purchase_ts).days
                if held_days < holding_period_days:
                    print(f"{position.symbol} held {held_days} days — within holding period.")
                    continue

                # 4) determine sell_price
                if market_open:
                    # 3) submit sell
                    print(f"ℹ️  Selling {position.symbol}, held {held_days} days.")
                    sell_order = submit_sell_order(client, position.symbol, position.qty)
                    
                    # poll for up to 5 minutes
                    deadline =  datetime.now(timezone.utc)+ timedelta(minutes=5)
                    sell_price = None
                    while datetime.now(timezone.utc) < deadline:
                        o = client.get_order(sell_order.id)
                        if o.filled_avg_price:
                            sell_price = float(o.filled_avg_price)
                            break
                        time.sleep(5)
                    if sell_price is None:
                        log_to_google_sheet(f"Sell to be executed but market is not responding")
                        print(f"⚠️  Sell for {position.symbol} not filled within 5 min. Falling back to last trade.")
                else:
                    print(f"⚠️  Market is closed.")
                    sell_price = None
                    log_to_google_sheet(f"Sell to be executed but market is closed")
            except Exception as e:
                print(f"Error processing {position.symbol}: {e}")

    except Exception as e:
        print(f"Error fetching open positions: {e}")
        
        
def place_order(client, symbol, amount, results_df):
    print("- Checking for stocks to buy...")
    order_placed = False
    # Get currently held positions
    held_symbols = {p.symbol for p in client.list_positions()}

    # Get open (unfilled) buy orders
    open_orders = client.list_orders(status='open')
    open_buy_symbols = {o.symbol for o in open_orders if o.side == 'buy'}

    if symbol in held_symbols:
        print(f"ℹ️  Skipping {symbol}: already held.")
        return order_placed

    if symbol in open_buy_symbols:
        print(f"ℹ️  Skipping {symbol}: buy order already open.")
        return order_placed

    try:
        price, qty_to_buy = get_price_and_quantity(client, symbol, amount)

        if qty_to_buy <= 0:
            print(f"ℹ️  Skipping {symbol}: ${amount:.2f} < ${price:.2f}")
            return order_placed

        print(f"✅  Placing BUY for {qty_to_buy} shares of {symbol} at ~${price:.2f} for a total of: ${qty_to_buy * price:.2f}.")
        submit_buy_order(client, symbol, qty_to_buy)
        order_placed = True
        
        log_message = f"Buy executed: {symbol}"
        if results_df is not None:
            details = get_fundamentals_and_prediction(symbol, results_df)
            log_message = f"{log_message}, {details}"

        log_to_google_sheet(log_message)

    except Exception as e:
        print(f"Error buying {symbol}: {e}")
    return order_placed