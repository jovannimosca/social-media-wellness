#!/bin/python

import pandas as pd
import numpy as np
import regex as re

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
   rawData = pd.read_csv('data/bigboy.csv')
   print('Done')
   
   # Manipulate the data:
   mentalHealthTags = ['mental[a-z ]*health', 'depress[a-z]*', 'anxiet[a-z]*']
   wellnessTags = ['[a-z ]*wellness[a-z ]*', '[a-z ]*health[a-z ]*', 'fitness', 'nutrition[a-z]*''sleep[a-z]*','mindful[a-z]*']
   filtered = filterHashtags(rawData, mentalHealthTags)
   print('Number of matching posts: ', len(filtered))
   filtered.to_csv('processed/relevant.csv')
   
   print()
   
   