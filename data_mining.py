import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw

class data:

    def __init__(self, start_date, end_date):
        self.start = start_date
        self.end = end_date
        quandl.ApiConfig.api_key = "m_zsbrMMZQBsSCT4_d3i"
    
    def oil_prices(self):
        self.price = quandl.get('OPEC/ORB', start_date=self.start, end_date=self.end)
        return self.price
    
    def oil_production(self):
        self.prod = quandl.get('JODI/OIL_CRPRKL_USA',start_date=self.start, end_date=self.end)
        return self.prod
    
    def oil_reserves(self):
        self.reserves = quandl.get('FED/IP_G211111C_N', start_date=self.start, end_date=self.end)
        return self.reserves
    
    def oil_imports(self):
        self.imports = quandl.get('EIA/PET_MTTIP_R10_ME0_2_M', start_date=self.start, end_date=self.end)
        return self.imports
    
    def gold_data(self):
        self.gold_prices = quandl.get('LBMA/GOLD', start_date=self.start, end_date=self.end)
        return self.gold_prices
    
    # def train_test_split(self, x, y, train_precent):
    #     train_precent = len(x) * train_precent
    #     train_X = x[:train_precent, :]
    #     train_y = y[:train_precent, :]
    #     test_x = x[train_precent:, :]
    #     test_y = y[train_precent:, :]

    #     return train_X, train_y, test_x, test_y