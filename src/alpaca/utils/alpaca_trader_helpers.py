from finvizfinance.quote import finvizfinance
import gspread
import json
import os
from google.oauth2.service_account import Credentials
from datetime import datetime, timezone
from dotenv import load_dotenv

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
        except:
            return "NA"

    try:
        stock = finvizfinance(ticker)
        info = stock.ticker_fundament()

        pe = parse_ratio(info.get("P/E"))
        ps = parse_ratio(info.get("P/S"))
        de = parse_ratio(info.get("Debt/Eq"))
    except Exception as e:
        pe, ps, de = "NA", "NA", "NA"

    try:
        pred_score = results_df.loc[results_df['Ticker'] == ticker, 'final_return_1m_raw_score'].values[0]
        pred_score = round(float(pred_score), 4)
    except Exception:
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