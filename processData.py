#!/bin/python

import pandas as pd
import numpy as np
import regex as re
import matplotlib.pyplot as plt
from nltk import *
from nltk.corpus import stopwords
import string

def filterHashtags(df:pd.DataFrame, hashtags:list[str]) -> pd.DataFrame:
   """Filter a pandas DataFrame based on a list of hashtags which may contain regex.

   Parameters
   ----------
   df : pd.DataFrame
       The dataframe to filter from.
   hashtags : list[str]
       A list of hashtags which may be regex.

   Returns
   -------
   pd.DataFrame
       A dataframe filtered by the list of hashtags provided.
   """
   tagRegex = "(,|$)|".join(["(^|,)"+tag for tag in hashtags])
   return df[df['hashtags'].apply(lambda row: re.search(tagRegex, str(row), flags=re.IGNORECASE) != None)]

if __name__ == '__main__':
   print('Reading from CSV...', end=' ')
   raw = pd.read_csv('data/bigboy.csv')
   ingested = raw[['id', 'full_text', 'hashtags', 'reply_count', 'user_name', 'likes', 'publicationtime']].convert_dtypes()
   ingested['publicationtime'] = pd.to_datetime(ingested['publicationtime'], format='%a %b %d %X %z %Y')
   print('Done')
   
   # Manipulate the data:
   mentalHealthTags = ['mental[a-z ]*health', 'depress[a-z]*', 'anxiet[a-z]*']
   wellnessTags = ['[a-z ]*wellness[a-z ]*', '[a-z ]*health[a-z ]*', 'fitness', 'nutrition[a-z]*''sleep[a-z]*','mindful[a-z]*']
   filtered = filterHashtags(ingested, wellnessTags)
   print('Number of matching tweets: ', len(filtered))
   
   # Let's see statistics by year:
   #   It looks like the vast majority of data is from 2021, so let's limit to that:
   grouped_by_month = filtered[filtered['publicationtime'].dt.year == 2021].groupby(pd.Grouper(key='publicationtime', freq='ME'))
   relevantPerMonth = grouped_by_month.size().reset_index(name='count')
   relevantPerMonth['month'] = relevantPerMonth['publicationtime'].dt.month_name()
   print(relevantPerMonth)
   graph = relevantPerMonth.plot(x='month', y='count', kind='bar', figsize=(6, 8))
   plt.savefig('img/relevantTweetsByMonth.png')
   plt.title('Number of Relevant Tweets by Month')
   # plt.show()
   
   # Tokenization:
   tt = TweetTokenizer()
   corpusTokens = set()
   for tweet in filtered['full_text']:
      for token in tt.tokenize(tweet):
         token = token.lower()
         if token not in stopwords.words('english') and token not in string.punctuation and not token.startswith('http') and not token.startswith('@'):
            corpusTokens.add(token)
            
   # Stemming:
   stemmer = PorterStemmer()
   stems = [stemmer.stem(token) for token in corpusTokens]
   
   # Frequency:
   wordlist = FreqDist(stems)
   word_features = [w for (w, c) in wordlist.most_common(30)]
   print('Top 30 Words: ', ", ".join(word_features))
   topFreq = FreqDist(word_features)
   topFreq.plot(title='Relevant Tweet Word Frequency')
   plt.savefig('img/relevantWordFrequency.png')
   
   # Save processed and filtered data:
   filtered.to_csv('processed/relevant.csv')
   