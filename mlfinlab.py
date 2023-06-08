import eikon as ek

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import mlfinlab

from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity

from mlfinlab.portfolio_optimization.mean_variance import MeanVarianceOptimisation

from mlfinlab.portfolio_optimization import ReturnsEstimators

from mlfinlab.portfolio_optimization import RiskEstimators

ek.set_app_key('e050640c60c44c4baa69bffc935b705d9c79b9a6')

import warnings

warnings.filterwarnings("ignore")

df,err = ek.get_data('MONITOR("Portfolio List 1")','TR.RIC') #or you could use a chain RIC like '0#.FTSE'

instruments = df['RIC'].astype(str).values.tolist()

start='2010-03-01'

end='2020-04-26'

ts = pd.DataFrame()

df = pd.DataFrame()


for r in instruments:

    try:

        ts = ek.get_timeseries(r,'CLOSE',start_date=start,end_date=end,interval='daily')

        ts.rename(columns = {'CLOSE': r}, inplace = True)

        if len(ts):

            df = pd.concat([df, ts], axis=1)

        else:

            df = ts

    except:

        pass

    df
































































































