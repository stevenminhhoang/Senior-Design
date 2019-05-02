import os
import click
import datetime
import pandas as pd
import fix_yahoo_finance as fy
import urllib.request


DIR = "data"

# Get stock data for S&P 500 companies from github repo "github.com/datasets/s-and-p-500-companies"
SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-companies-financials/master/data/constituents-financials.csv"
SP500_PATH = os.path.join(DIR, "constituents-financials.csv")


def get_sp500():
    if os.path.exists(SP500_PATH):
        return

    response = urllib.request.urlopen(SP500_URL)
    text = response.read().decode("utf-8")
    print("Downloading ...", SP500_URL)
    with open(SP500_PATH, "w") as fin:
        print(text, file=fin)
    return


def get_individual_stock(symbol, start_date, end_date):
    stock = fy.download(symbol, start_date, end_date)
    stock_data = stock.to_csv(os.path.join(DIR, "%s.csv" % symbol))
    return


def load_sp500():
    get_sp500()
    df_sp500 = pd.read_csv(SP500_PATH)
    df_sp500.sort_values(by="Market Cap", ascending=False, inplace=True)
    stock_symbols = df_sp500["Symbol"].unique().tolist()
    return stock_symbols


@click.command()
@click.option("--symbol", help="Stock symbol to get latest prices")
def main(symbol):
    data = load_sp500()
    now = datetime.datetime.now()
    curr_date = now.strftime("%Y-%m-%d")
    get_individual_stock(symbol, "2010-01-01", curr_date)

if __name__ == "__main__":
    main()
