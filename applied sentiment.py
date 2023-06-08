import eikon as ek
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from textblob import TextBlob
import datetime
#from datetime import time
import warnings
warnings.filterwarnings("ignore")
ek.set_app_key('b45540b94f0a4a9cbe7a76c868f905063418501c')

df1 = ek.get_news_headlines('NBR.N', date_from= "2021-10-14",
                           count=100)
df2 = ek.get_news_headlines('TLRY.O', date_from= "2021-10-14",
                            count=100)
df1.head()
df1['versionCreated'] = df1['versionCreated'].dt.strftime('%Y-%m-%d %H:%M:%S')
df1['date'] = pd.to_datetime(df1['versionCreated']).dt.date
df1['time'] = pd.to_datetime(df1['versionCreated']).dt.time
df2['versionCreated'] = df2['versionCreated'].dt.strftime('%Y-%m-%d %H:%M:%S')
df2['date'] = pd.to_datetime(df2['versionCreated']).dt.date
df2['time'] = pd.to_datetime(df2['versionCreated']).dt.time

ticker1="NBR"
df1["ticker"]=ticker1
ticker2="TLRY"
df2["ticker"]=ticker2
df=df1.append(df2)
# NLTK VADER for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# New words and values
new_words = {
    'crushes': 10,
    'beats': 5,
    'misses': -5,
    'trouble': -10,
    'falls': -100,
}

import nltk
nltk.downloader.download('vader_lexicon')

# Instantiate the sentiment intensity analyzer with the existing lexicon
vader = SentimentIntensityAnalyzer()
# Update the lexicon
vader.lexicon.update(new_words)
# Iterate through the headlines and get the polarity scores
scores = [vader.polarity_scores(text) for text in df.text]
# Convert the list of dicts into a DataFrame
scores_df = pd.DataFrame(scores)
df = df.reset_index()
# Join the DataFrames
scored_news = df.join(scores_df)


import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
#%matplotlib inline

# Group by date and ticker columns from scored_news and calculate the mean
mean_c = scored_news.groupby(['date', 'ticker']).mean()
# Unstack the column ticker
mean_c = mean_c.unstack('ticker')
# Get the cross-section of compound in the 'columns' axis
mean_c = mean_c.xs("compound", axis="columns")
# Plot a bar chart with pandas
mean_c.plot.bar(figsize = (10, 6));    
    
# Count the number of headlines in scored_news (store as integer)
num_news_before = scored_news.text.count()
# Drop duplicates based on ticker and headline
scored_news_clean = scored_news.drop_duplicates(subset=['text', 'ticker'])
# Count number of headlines after dropping duplicates (store as integer)
num_news_after = scored_news_clean.text.count()
# Print before and after numbers to get an idea of how we did 
f"Before we had {num_news_before} headlines, now we have {num_news_after}"    
    
# Set the index to ticker and date
single_day = scored_news_clean.set_index(['date', 'ticker'])
single_day.sort_index(ascending=True, inplace=True)
# Select the 3rd of January of 2019
single_day = single_day.loc['2021-10-19']
single_day = pd.DataFrame(single_day)

# Convert the datetime string to just the time
#single_day["time"] = pd.to_datetime(single_day['time']).dt.time
# Set the index to time and sort by it
single_day = single_day.set_index("time")
# Sort it
single_day = single_day.sort_index()    
    
TITLE = "Negative, neutral, and positive sentiment for NBR on 2021-10-19"
COLORS = ["red","orange", "green"]
# Drop the columns that aren't useful for the plot
plot_day = single_day.drop(['compound', 'text',"index","versionCreated","storyId","sourceCode"], 1)
# Change the column names to 'negative', 'positive', and 'neutral'
plot_day.columns = ['negative', 'neutral', 'positive']
# Plot a stacked bar chart
plot_day.plot.bar(stacked = True, figsize=(10, 6), title = TITLE, color = COLORS).legend(bbox_to_anchor=(1.2, 0.5))
plt.ylabel("scores");    
    
    
    
    