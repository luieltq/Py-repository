import os

os.chdir(r"C:\Users\luiel\OneDrive\Documentos\data")

import sys

sys.path.append(r"C:\Users\luiel\OneDrive\Documentos\data\data")

print(sys.path)

import numpy as np

prices_a = np.array([8.70, 8.91, 8.71])
prices_a

prices_a[1:]/prices_a[:-1] - 1

import pandas as pd

prices = pd.DataFrame({"BLUE": [8.70, 8.91, 8.71, 8.43, 8.73],
                       "ORANGE": [10.66, 11.08, 10.71, 11.59, 12.11]})

prices.iloc[1:]

prices.iloc[:-1]
#error
prices.iloc[1:]/prices.iloc[:-1] - 1
#salidas al error
prices.iloc[1:].values/prices.iloc[:-1] - 1

prices.iloc[1:]/prices.iloc[:-1].values - 1

prices

prices.shift(1)

returns = prices/prices.shift(1) - 1
returns

returns = prices.pct_change()
returns

prices = pd.read_csv('data/sample_prices.csv')
prices
#algunos estadísticos
returns = prices.pct_change()
returns

returns.mean()

returns.std()
#ploteando
returns.plot.bar()

prices.plot()
#calculando retorno
np.prod(returns+1)

(returns+1).prod()

(returns+1).prod()-1

(((returns+1).prod()-1)*100).round(2)

#anualizando retorno mensual
rm = 0.01
(1+rm)**12 - 1
#anualizando retorno trimestral
rq = 0.04
(1+rq)**4 - 1

#VOLATILITY AND RISK

import pandas as pd
prices = pd.read_csv("data/sample_prices.csv")
returns = prices.pct_change()
returns

returns = returns.dropna()
returns

deviations = returns - returns.mean()
squared_deviations = deviations**2
mean_squared_deviations = squared_deviations.mean()

import numpy as np
#population standard deviation
volatility = np.sqrt(mean_squared_deviations)
volatility
#sample standard deviation
returns.std()

#numer of row and columns
returns.shape

number_of_obs = returns.shape[0]
mean_squared_deviations = squared_deviations.sum()/(number_of_obs-1)
volatility = np.sqrt(mean_squared_deviations)
volatility

returns.std()

#numer of row and columns
annualized_vol = returns.std()*(12**0.5)
annualized_vol

me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                   header=0, index_col=0, parse_dates=True, na_values=-99.99)
me_m.head()

cols = ['Lo 10', 'Hi 10']
returns = me_m[cols]
returns.head()

returns = returns/100

returns.plot()

returns.columns = ['SmallCap', 'LargeCap']

returns.plot()

annualized_vol = returns.std()*np.sqrt(12)
annualized_vol

n_months = returns.shape[0]
return_per_month = (returns+1).prod()**(1/n_months) - 1
return_per_month

annualized_return = (return_per_month + 1)**12-1

annualized_return = (returns+1).prod()**(12/n_months) - 1
annualized_return

annualized_return/annualized_vol

riskfree_rate = 0.03
excess_return = annualized_return - riskfree_rate
sharpe_ratio = excess_return/annualized_vol
sharpe_ratio

#Computing Maximum Drawdown

import pandas as pd

me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                   header=0, index_col=0, parse_dates=True, na_values=-99.99)

rets = me_m[['Lo 10', 'Hi 10']]
rets.columns = ['SmallCap', 'LargeCap']
rets = rets/100
rets.plot.line()

rets.index

rets.index = pd.to_datetime(rets.index, format="%Y%m")
rets.index

rets["2008"]

rets.index = rets.index.to_period('M')
rets.head()

rets.info()

rets.describe()

wealth_index = 1000*(1+rets["LargeCap"]).cumprod()
wealth_index.plot()

previous_peaks = wealth_index.cummax()
previous_peaks.plot()

drawdown = (wealth_index - previous_peaks)/previous_peaks
drawdown.plot()

drawdown.min()

drawdown["1975":].plot()

drawdown["1975":].min()

def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})

drawdown(rets["LargeCap"]).head()

drawdown(rets["LargeCap"]).min()

drawdown(rets["SmallCap"]).min()

drawdown(rets["LargeCap"])["Drawdown"].idxmin()

drawdown(rets["SmallCap"])["Drawdown"].idxmin()

drawdown(rets["LargeCap"]["1975":])["Drawdown"].idxmin()

drawdown(rets["SmallCap"]["1975":])["Drawdown"].idxmin()

drawdown(rets["SmallCap"]["1975":])["Drawdown"].min()


#Deviations from Normality

def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                      header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

%load_ext autoreload
%autoreload 2

import pandas as pd
import edhec_risk_kit_105 as erk
hfi = erk.get_hfi_returns()
hfi.head()

pd.concat([hfi.mean(), hfi.median(), hfi.mean()>hfi.median()], axis=1)

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

skewness(hfi).sort_values()

import scipy.stats
scipy.stats.skew(hfi)

hfi.shape

import numpy as np
normal_rets = np.random.normal(0, 0.15, (263, 1))

normal_rets.mean(), normal_rets.std()

erk.skewness(normal_rets)

def kurtosis(r):
    """
    Alternative to scipy.stats.skew()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

erk.kurtosis(hfi)

scipy.stats.kurtosis(hfi)

scipy.stats.kurtosis(normal_rets)

erk.kurtosis(normal_rets)

scipy.stats.jarque_bera(normal_rets)

scipy.stats.jarque_bera(hfi)

erk.is_normal(normal_rets)

hfi.aggregate(erk.is_normal)

import scipy.stats
def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level

import pandas as pd
isinstance(hfi, pd.DataFrame)

erk.is_normal(normal_rets)

ffme = erk.get_ffme_returns()
erk.skewness(ffme)

erk.kurtosis(ffme)

erk.is_normal(ffme)

#Downside Measures: SemiDeviation, VaR and CVaR

import pandas as pd
import edhec_risk_kit_106 as erk
%load_ext autoreload
%autoreload 2
%matplotlib inline

hfi = erk.get_hfi_returns()

def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

erk.semideviation(hfi)

hfi[hfi<0].std(ddof=0)

erk.semideviation(hfi).sort_values()

ffme = erk.get_ffme_returns()
erk.semideviation(ffme)

def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

import numpy as np
np.percentile(hfi,50)

erk.var_historic(hfi, level=1)

erk.cvar_historic(hfi, level=1).sort_values()

erk.cvar_historic(ffme)

from scipy.stats import norm
norm.ppf(.5)

norm.ppf(.16)

erk.var_gaussian(hfi)

erk.var_historic(hfi)

from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )

    return -(r.mean() + z*r.std(ddof=0))

var_table = [erk.var_gaussian(hfi), 
             erk.var_gaussian(hfi, modified=True), 
             erk.var_historic(hfi)]
comparison = pd.concat(var_table, axis=1)
comparison.columns=['Gaussian', 'Cornish-Fisher', 'Historic']
comparison.plot.bar(title="Hedge Fund Indices: VaR at 5%")

erk.skewness(hfi).sort_values(ascending=False)












