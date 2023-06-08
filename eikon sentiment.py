import eikon as ek
import numpy as np
from bs4 import BeautifulSoup
from textblob import TextBlob
import datetime
#from datetime import time
import warnings
warnings.filterwarnings("ignore")
ek.set_app_key('b45540b94f0a4a9cbe7a76c868f905063418501c')

df = ek.get_news_headlines('TLRY.O AND Language:LEN', date_to = "2020-12-04", count=100)
df.head()

df['Polarity'] = np.nan
df['Subjectivity'] = np.nan
df['Score'] = np.nan

for idx, storyId in enumerate(df['storyId'].values):  #for each row in our df dataframe
    newsText = ek.get_news_story(storyId) #get the news story
    if newsText:
        soup = BeautifulSoup(newsText,"lxml") #create a BeautifulSoup object from our HTML news article
        sentA = TextBlob(soup.get_text()) #pass the text only article to TextBlob to anaylse
        df['Polarity'].iloc[idx] = sentA.sentiment.polarity #write sentiment polarity back to df
        df['Subjectivity'].iloc[idx] = sentA.sentiment.subjectivity #write sentiment subjectivity score back to df
        if sentA.sentiment.polarity >= 0.05: # attribute bucket to sentiment polartiy
            score = 'positive'
        elif  -.05 < sentA.sentiment.polarity < 0.05:
            score = 'neutral'
        else:
            score = 'negative'
        df['Score'].iloc[idx] = score #write score back to df
df.head()

start = df['versionCreated'].min().replace(hour=0,minute=0,second=0,microsecond=0).strftime('%Y/%m/%d')
end = df['versionCreated'].max().replace(hour=0,minute=0,second=0,microsecond=0).strftime('%Y/%m/%d')
Minute = ek.get_timeseries([".NDX"], start_date=start, interval="minute")
Minute.tail()

df['twoM'] = np.nan
df['fiveM'] = np.nan
df['tenM'] = np.nan
df['thirtyM'] = np.nan
df.head(2)

for idx, newsDate in enumerate(df['versionCreated'].values):
    sTime = df['versionCreated'][idx]
    sTime = sTime.replace(second=0,microsecond=0)
    try:
        t0 = Minute.iloc[Minute.index.get_loc(sTime),2]
        df['twoM'][idx] = ((Minute.iloc[Minute.index.get_loc((sTime + datetime.timedelta(minutes=2))),3]/(t0)-1)*100)
        df['fiveM'][idx] = ((Minute.iloc[Minute.index.get_loc((sTime + datetime.timedelta(minutes=5))),3]/(t0)-1)*100)
        df['tenM'][idx] = ((Minute.iloc[Minute.index.get_loc((sTime + datetime.timedelta(minutes=10))),3]/(t0)-1)*100) 
        df['thirtyM'][idx] = ((Minute.iloc[Minute.index.get_loc((sTime + datetime.timedelta(minutes=30))),3]/(t0)-1)*100)
    except:
        pass
df.head()

grouped = df.groupby(['Score']).mean()
grouped







































































