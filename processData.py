#!/bin/python

import pandas as pd
# import numpy as np
import regex as re
# import matplotlib.pyplot as plt
# from nltk import *
# from nltk.corpus import stopwords
# import string
# from wordcloud import WordCloud

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
   print(f'Total number of tweets: {len(ingested)}')
   
   # Manipulate the data:
   mentalHealthTags = ['mental[a-z ]*health', 'depress[a-z]*', 'anxiet[a-z]*']
   wellnessTags = ['[a-z ]*wellness[a-z ]*', '[a-z ]*health[a-z ]*', r'fitness\d*', 'nutrition[a-z]*', 'sleep[a-z]*', 'mindful[a-z]*', r'diet\d*', r'\d*workout\d*', 'HIIT', r'\d*fasting']
   filtered = filterHashtags(ingested, wellnessTags)
   print(f'Number of matching tweets: {len(filtered)} ({len(filtered)/len(ingested)*100:.2f}%)')
   
   # Save processed and filtered data:
   ingested.to_csv('processed/cleaned.csv')
   filtered.to_csv('processed/relevant.csv')
   print('Wrote data to processed/')
   