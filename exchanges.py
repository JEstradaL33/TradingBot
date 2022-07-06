"""
Classes for connecting to different cryptocurrency exchanges and extracting market data via APIs.
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime

class BybitExch():

    """
    Class for interacting with Bybit exchange.
    https://www.bybit.com

    Actual endpoint: https://api.bybit.com/v2/
    """
    def __init__(self, endpoint, api_key = None, private_key = None):

        self.endpoint = endpoint
        self.api_key = api_key
        self.private_key = private_key

    def server_time(self):
        time_endpoint = '/v2/public/time'
        server_time = int(float(requests.get(url = self.endpoint + time_endpoint).json()['time_now']))
        return server_time

    def extract_ohlcv(self, start, end, symbol, interval, limit = 200):

        """
        Exctracts historical OHLCV data for a given symbol. All timestamps are converted to UTC.
        Parameters:
        - start
            -- int, timestamp
            -- extract data since timestamp
        - end
            -- int, timestamp
            -- extract data until timestamp
        - interval
            -- int, timeframe in seconds
            -- available_values, {1 3 5 15 30 60 120 240 360 720 1440 10080 43200}
        - symbol
            -- string
            -- E.g. 'BTCUSD'
        - limit
            -- int, number of candles per request
            -- max available is 200
        Returns:
        - pd.DataFrame
        """

        n_candles = (end - start) / interval
        n_intervals = int(n_candles / limit) + 1
        intervals = [end] + [end - (x * interval) for x in range(limit, limit * (n_intervals + 1), limit)]
        output = pd.DataFrame()

        for element in intervals:
            params = {'from': element, 'interval': interval, 'symbol': symbol}
            result = pd.DataFrame(requests.get(url = self.endpoint + '/v2/public/kline/list', params = params).json()['result'])
            output = pd.concat([output, result], axis = 0)
            time.sleep(0.4)

        output = output.loc[:, ['open_time', 'low', 'high', 'open', 'close', 'volume']]
        output = output.loc[:, ['Time', 'Low', 'High', 'Open', 'Close', 'Volume']]
        output = output.set_index('Time', inplace = True)
        output.sort_index(ascending = True, inplace = True)
        for col in output.columns:
            output[col] = pd.to_numeric(output[col])

        return output

class CoinbaseProExch():

    """
    Class for interacting with Coinbase Pro exchange.
    https://www.pro.coinbase.com

    Actual endpoint: 'https://api.pro.coinbase.com/'
    """

    def __init__(self, endpoint, api_key = None, private_key = None):

        self.endpoint = endpoint
        self.api_key = api_key
        self.public_key = private_key

    def extract_ohlcv(self, start, end, granularity, symbol, limit = 300):

        """
        Exctracts historical OHLCV data for a given symbol
        Parameters:
        - start
            -- int, timestamp
            -- lower range value
        - end
            -- int, timestamp
            -- upper range value
        - granularity
            -- int, timeframe in seconds
            -- available_values, {60, 300, 900, 3600, 21600, 86400}
        - symbol
            -- string
            -- E.g. 'BTCUSD'
        - n_candles
            -- int, number of candles per request
            -- max available is 300
        Returns:
        - pd.DataFrame
        """

        n_candles = (end - start) / granularity
        n_intervals = int(n_candles / limit) + 1
        intervals = [end] + [end - (x * granularity) for x in range(limit, limit * (n_intervals + 1), limit)]
        output = pd.DataFrame()

        for i in range(1, len(intervals)):
            params = {'start': datetime.fromtimestamp(intervals[i]).isoformat(), 'end': datetime.fromtimestamp(intervals[i - 1]).isoformat(), 'granularity': granularity}
            result = pd.DataFrame((requests.get(url = self.endpoint + 'products/' + symbol + '/candles', params = params).json()))
            output = pd.concat([output, result], axis = 0)
            time.sleep(0.2)

        output.columns = ['Time', 'Low', 'High', 'Open', 'Close', 'Volume']
        output.set_index('Time', inplace = True, drop = True)
        output.sort_index(ascending = True, inplace = True)
        output = output[~output.index.duplicated(keep='first')]

        for col in output.columns:
            output[col] = pd.to_numeric(output[col])

        return output

class BitfinexExch():

    """
    Class for interacting with Bitfinex exchange.
    https://www.bitfinex.com

    Actual endpoint: https://api-pub.bitfinex.com/v2/
    """

    def __init__(self, endpoint, api_key = None, private_key = None):

        self.endpoint = endpoint
        self.api_key = api_key
        self.public_key = private_key

    def extract_ohlcv(self, timeframe, symbol, section, start, end, limit = 10000, period = None):

        """
        Exctracts historical OHLCV data for a given symbol. All timestamps are converted to UTC.
        Parameters:
        - timeframe
            -- string
            -- available_values: '1m', '5m', '15m', '30m', '1h', '3h', '6h', '12h', '1D', '7D', '14D', '1M'
        - symbol
            -- string
            -- E.g.: 'tBTCUSD'
            -- t for trading pair, f for funding
            -- to fetch available tickers use Configs endpoint
        - section
            -- string
            -- available_values: 'last' and 'hist'
        - period
            -- string
            -- Only required for funding candles
        - start
            -- int, timestamp
            -- lower range value
        - end
            -- int, timestamp
            -- upper range value
        - limit
            -- int, maximum number of records returned by each query

        Returns:
        - pd.DataFrame
        """

        secs_in_tf = {'1m': 60, '5m': 300, '15m': 900, '30m': 1800, '1h': 3600, '3h': 10800,
                    '6h': 21600, '12h': 43200, '1D': 86400, '7D': 604800, '14D': 1209600, '1M': 2592000}
        n_candles = (end - start) / secs_in_tf[timeframe]
        n_intervals = int(n_candles / limit) + 1
        intervals = [end] + [end - (x * secs_in_tf[timeframe]) for x in range(limit, limit * (n_intervals + 1), limit)]
        output = pd.DataFrame()

        if period is None:
            path_param = ':' + timeframe + ':' + symbol + '/' + section
        else:
            path_param = ':' + timeframe + ':' + symbol + ':' + period + '/' + section

        for i in range(1, len(intervals)):
            params = {'limit': limit, 'start': str(intervals[i]) + '000', 'end': str(intervals[i - 1]) + '000'}
            result = pd.DataFrame((requests.get(url = self.endpoint + '/candles/trade' + path_param, params = params).json()))
            output = pd.concat([output, result], axis = 0)
            time.sleep(0.8)


        output.columns = ['Time', 'Open', 'Close', 'High', 'Low', 'Volume']
        output.set_index('Time', inplace = True, drop = True)
        output.sort_index(ascending = True, inplace = True)
        output = output[~output.index.duplicated(keep='first')]

        for col in output.columns:
            output[col] = pd.to_numeric(output[col])

        return output
