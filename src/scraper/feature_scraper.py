import os
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
from datetime import datetime, timedelta

from .utils.feature_scraper_helpers import *
from .utils.technical_indicators_helpers import *
from .utils.financial_ratios_helpers import *
from src.alpaca.utils.alpaca_trader_helpers import log_to_google_sheet

class FeatureScraper:
    def __init__(self):
        self.base_url = "http://openinsider.com/screener?"
        self.data = pd.DataFrame()
        
    def process_web_page(self, date_range):
        start_date, end_date = date_range
        url = f"{self.base_url}pl=1&ph=&ll=&lh=&fd=-1&fdr={start_date.month}%2F{start_date.day}%2F{start_date.year}+-+{end_date.month}%2F{end_date.day}%2F{end_date.year}&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&vl=10&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=1000&page=1"
        return fetch_and_parse(url)

    def fetch_data_from_pages(self, num_days):
        end_date = datetime.datetime.now()
        date_ranges = []

        # Prepare the date ranges
        for _ in range(num_days):
            start_date = end_date - datetime.timedelta(days=1)  # Each range is 1 month
            date_ranges.append((start_date, end_date))
            end_date = start_date  # Move back another month

        # Use multiprocessing to fetch and parse data in parallel
        with Pool(cpu_count()) as pool:
            data_frames = list(
                tqdm(
                    pool.imap(self.process_web_page, date_ranges),
                    total=len(date_ranges),
                    desc="- Scraping entries from openinsider.com for last week"
                )
            )

        # Filter out None values (pages where no valid table was found)
        data_frames = [df for df in data_frames if df is not None]

        if data_frames:
            self.data = pd.concat(data_frames, ignore_index=True)
            print(f"- {len(self.data)} total entries extracted!")
        else:
            print(f"🚫 No trades were made in the past {num_days} days")
    
    def clean_table(self, drop_threshold=0.05):
        columns_of_interest = ["Filing Date", "Trade Date", "Ticker", "Title", "Price", "Qty", "Owned", "ΔOwn", "Value"]
        self.data = self.data[columns_of_interest]
        self.data = process_dates(self.data)
        
        # Filter out entries where Filing Date is less than 20 business days in the past
        cutoff_date = pd.to_datetime('today')
        self.data = self.data[self.data['Filing Date'] < cutoff_date]
        
        # Clean numeric columns
        self.data = clean_numeric_columns(self.data)
        
        # Drop rows where ΔOwn is negative
        self.data = self.data[self.data['ΔOwn'] >= 0]
        
        # Parse titles
        self.data = parse_titles(self.data)
        self.data.drop(columns=['Title', 'Trade Date'], inplace=True)
        
        # Show the number of unique Ticker - Filing Date combinations
        unique_combinations = self.data[['Ticker', 'Filing Date']].drop_duplicates().shape[0]
        print(f"- Number of unique Ticker - Filing Date combinations before aggregation: {unique_combinations}")
        
        # Group by Ticker and Filing Date, then aggregate
        self.data = aggregate_group(self.data)
        
        # Format the date column and drop any remaining rows with missing values
        self.data['Filing Date'] = self.data['Filing Date'].dt.strftime('%d-%m-%Y %H:%M')
        
        # Clean the data by dropping columns with more than 5% missing values and then dropping rows with missing values
        self.data = clean_data(self.data, drop_threshold)

        
    def add_technical_indicators(self, drop_threshold=0.05):
        rows = self.data.to_dict('records')
        
        # Apply technical indicators
        with Pool(cpu_count()) as pool:
            processed_rows = list(
                tqdm(
                    pool.imap(process_ticker_technical_indicators, rows),
                    total=len(rows),
                    desc="- Scraping technical indicators"
                )
            )
        
        self.data = pd.DataFrame(filter(None, processed_rows))
        
        # Replace infinite values and drop rows with missing values
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Clean the data by dropping columns with more than 5% missing values and then dropping rows with missing values
        self.data = clean_data(self.data, drop_threshold)


    def add_financial_ratios(self, drop_threshold=0.05):
        """
        Scrape financial ratios in parallel using all CPUs,
        then expand 'Sector' into dummies and clean missing data.
        """
        rows = self.data.to_dict('records')
        
        # Parallelize the I/O‐bound financial‐ratio scraping
        with Pool(cpu_count()) as pool:
            processed_rows = list(
                tqdm(
                    pool.imap(process_ticker_financial_ratios, rows),
                    total=len(rows),
                    desc="- Scraping financial ratios"
                )
            )
        
        # Drop any Nones and rebuild DataFrame
        self.data = pd.DataFrame(filter(None, processed_rows))
        
        # If there's a Sector column, one‐hot encode it
        if 'Sector' in self.data.columns:
            sector_dummies = pd.get_dummies(
                self.data['Sector'], prefix='Sector', dtype=int
            )
            # drop original and concat dummies
            self.data = pd.concat(
                [self.data.drop(columns=['Sector']), sector_dummies],
                axis=1
            )
        
        # Replace any infinite values and clean by threshold + dropping NaNs
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data = clean_data(self.data, drop_threshold)


    def add_insider_transactions(self, drop_threshold=0.05):
        rows = self.data.to_dict('records')
        tickers = [row['Ticker'] for row in rows]
        # Fetch insider transactions
        with Pool(cpu_count()) as pool:
            processed_rows = list(
                tqdm(
                    pool.imap(get_recent_trades, tickers),
                    total=len(tickers),
                    desc="- Scraping recent insider trades"
                )
            )
        
        for row, trade_data in zip(rows, processed_rows):
            if trade_data:
                row.update(trade_data)
        
        self.data = pd.DataFrame(rows)
        
        # Clean the data by dropping columns with more than 5% missing values and then dropping rows with missing values
        self.data = clean_data(self.data, drop_threshold)
    
            
    def load_sheet(self, file_path='output.xlsx'):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        file_path = os.path.join(data_dir, file_path)

        if os.path.exists(file_path):
            try:
                self.data = pd.read_excel(file_path)
                print(f"- Sheet successfully loaded from {file_path}.")
            except Exception as e:
                print(f"- Failed to load sheet from {file_path}: {e}")
        else:
            print(f"- File '{file_path}' does not exist.")
        
    def run(self, num_days):
        start_time = time.time()
        print("\n### START ### Feature Scraper")
        self.fetch_data_from_pages(num_days)
        if self.data.empty:
            log_to_google_sheet(f"No trades were made in the past {num_days} days")
            return pd.DataFrame()
        self.clean_table(drop_threshold=0.05)
        self.add_technical_indicators(drop_threshold=0.05)
        self.add_financial_ratios(drop_threshold=0.2)
        self.add_insider_transactions(drop_threshold=0.05)
        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Feature Scraper - time elapsed: {elapsed_time}")
        return self.data
    
