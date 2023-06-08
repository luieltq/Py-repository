import eikon as ek

import pandas

import numpy

import matplotlib.pyplot as plt

import scipy.optimize as sco

import os

ek.set_app_key('e050640c60c44c4baa69bffc935b705d9c79b9a6')

N=100

constituents, err = ek.get_data(['IWDA.L'], ['TR.ETPConstituentRIC', 'TR.ETPConstituentName'])

constituents.rename(columns={'Constituent RIC': 'ric', 'Constituent Name': 'name'}, inplace=True)

constituents = constituents[['ric','name']][0:N]

print(constituents)

start='2020-01-01'

end='2020-12-31'

instruments = constituents['ric'].astype(str).values.tolist()

ts =pandas.DataFrame()

for r in instruments:

    try:

        ts1 = ek.get_timeseries(r,'CLOSE',start_date=start,end_date=end,interval='daily')

        ts1.rename(columns = {'CLOSE' : r}, inplace=True)

        ts =pandas.concat([ts, ts1], axis=1)

    except:

        continue

ts = ts.dropna()

print(ts)

df_esg, err = ek.get_data(instruments, ['TR.TRESGScore','TR.BusinessSummary'])

df_esg = df_esg.rename(columns={'Instrument':'ric', 'ESG Score':'esg'}).set_index('ric')

df_esg = pandas.concat([constituents.set_index('ric'), df_esg], join='inner', axis=1)

df_esg

returns = ts.pct_change().replace(numpy.inf, numpy.nan).dropna()

covMatrix = returns.cov()

def risk_measure(covMatrix, weights):

    return numpy.dot(weights, numpy.dot(covMatrix, weights))

constraints = {'type': 'eq', 'fun': lambda weights: weights.sum() - 1}

mvp = sco.minimize(lambda x: risk_measure(covMatrix, x), 
                   len(instruments) * [1 / len(instruments)], 
                   bounds=None,
                   constraints =constraints
                   )

mvp_weights = list(mvp['x'])

mvp_esg = numpy.dot(mvp_weights, df_esg['esg'])

mvp_risk = mvp['fun']

print('\nMVP in a universe with {N} instruments\nNumber of selected instruments: {n}\nMinimum weight: {minw}\nMaximum weight: {maxw}\nHistorical risk measure: {risk}\nHistorical return p.a.: {r}\nESG score: {esg}'.format(N=N,n=numpy.sum(mvp['x']>0),minw=numpy.min(mvp['x'][numpy.nonzero(mvp['x'])]),maxw=numpy.max(mvp['x']),risk=mvp_risk,r=numpy.dot(mvp_weights,returns.sum()),esg=mvp_esg))

df_weights =pandas.DataFrame(data=mvp_weights, index=instruments)

df_weights =df_weights.sort_values(by=[0], ascending=False)

df_weights =df_weights[df_weights > 1e-4].dropna()

df_weights =df_weights.T

#plotting the weights 

plt.figure(figsize=(15, 5))

ypos = numpy.linspace(0, 100, num=len(df_weights.iloc[0,:]))

plt.bar(ypos, df_weights.values[0], width =1)

plt.xticks(ypos, df_weights.columns, rotation =30, ha ='right')

plt.xlabel('RICs', fontsize=12)

plt.ylabel('Weights', fontsize=12)

plt.title('Portfolio Weights (Minimum Volatility Portfolio)', fontsize=12)

plt.show()

prescribed_esg = 80


esg_constraint = {'type': 'eq', 'fun': lambda weights: numpy.dot(weights, df_esg['esg']) - prescribed_esg}

esgmvp = sco.minimize(lambda x: risk_measure(covMatrix, x), 
                      len(instruments) * [1 / len(instruments)], 
                      bounds=None,
                      constraints =[constraints, esg_constraint]
                      )

esgmvp_weights = list(esgmvp['x'])

esgmvp_esg = numpy.dot(esgmvp_weights, df_esg['esg'])

esgmvp_risk = esgmvp['fun']


print('\nMVP with prescribes ESG = {pe} in a universe with {N} instruments\nNumber of selected instruments: {n}\nMinimum weight: {minw}\nMaximum weight: {maxw}\nHistorical risk measure: {risk}\nHistorical return p.a.: {r}\nESG score: {esg}'.format(N=N,n=numpy.sum(esgmvp['x']>0),minw=numpy.min(esgmvp['x'][numpy.nonzero(esgmvp['x'])]),maxw=numpy.max(esgmvp['x']),risk=esgmvp_risk,r=numpy.dot(esgmvp_weights,returns.sum()),esg=esgmvp_esg,pe=prescribed_esg))

df_weights =pandas.DataFrame(data=esgmvp_weights, index=instruments)

df_weights =df_weights.sort_values(by=[0], ascending=False)

df_weights =df_weights[df_weights > 1e-4].dropna()

df_weights =df_weights.T

#plotting the weights

plt.figure(figsize=(15, 5))

ypos = numpy.linspace(0, 100, num=len(df_weights.iloc[0,:]))

plt.bar(ypos, df_weights.values[0], width =1)

plt.xticks(ypos, df_weights.columns, rotation =30, ha ='right')

plt.xlabel('RICs', fontsize=12)

plt.ylabel('Weights', fontsize=12)

plt.title('Portfolio Weights (Minimal Risk with ESG Score 80)', fontsize=12)

plt.show()

max_esg=numpy.floor(max(df_esg['esg'].tolist())) #max esg value to be achieved dependent on the universe

min_esg=mvp_esg #min esg value which is interesting for portfolio selection


results = {'esg_val':[],'weights':[],'risk':[],'return':[]}

for rho in numpy.linspace(min_esg,max_esg,num=25):

    constraints2 = {'type': 'eq', 'fun': lambda weights: numpy.dot(weights, df_esg['esg']) - rho}

    res = sco.minimize(lambda x: risk_measure(covMatrix, x),
                       len(instruments) * [1 / len(instruments)],
                       bounds=None,
                       constraints =[constraints, constraints2]
                       )

weights = list(res['x'])

esg_val= numpy.dot(weights, df_esg['esg'])

results['esg_val'].append(esg_val)

results['weights'].append(weights)

results['risk'].append(res['fun'])

results['return'].append(numpy.dot(weights,returns.sum()))

plt.plot(results['risk'],results['esg_val'], 'o')

plt.xlabel('risk')

plt.ylabel('esg_val')

plt.show()

plt.plot(results['esg_val'],results['return'],'o')

plt.xlabel('esg_val')

plt.ylabel('return')

plt.show()





























































































