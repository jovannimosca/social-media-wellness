import pandas as pd
import matplotlib.pyplot as plt
from nltk import *
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud

if __name__ == '__main__':
   # Get processed data:
   print('Reading from CSV...', end=' ')
   ingested = pd.read_csv('processed/cleaned.csv', parse_dates=['publicationtime']).convert_dtypes()
   filtered = pd.read_csv('processed/relevant.csv', parse_dates=['publicationtime']).convert_dtypes()
   print(f'Total number of tweets: {len(ingested)}')
   print(f'Number of filtered tweets: {len(filtered)} ({len(filtered)/len(ingested)*100:.2f}%)')
   
   # Let's see statistics by year:
   monthly = ingested.groupby(pd.Grouper(key='publicationtime', freq='ME'))
   monthly = monthly.size().reset_index(name='count')
   monthly['month'] = monthly['publicationtime'].dt.month_name()
   monthly.plot(x='month', y='count', kind='line', figsize=(8, 6))
   plt.title('Number of Tweets by Month')
   plt.savefig('img/allTweetsByMonth.png')
   #   It looks like the vast majority of data is from 2021, so let's limit to that:
   grouped_by_month = filtered[filtered['publicationtime'].dt.year == 2021].groupby(pd.Grouper(key='publicationtime', freq='ME'))
   relevantPerMonth = grouped_by_month.size().reset_index(name='count')
   relevantPerMonth['month'] = relevantPerMonth['publicationtime'].dt.month_name()
   print('Relevant Tweets by Month:\n', relevantPerMonth[['month', 'count']], end='\n\n')
   relevantPerMonth.plot(x='month', y='count', kind='line', figsize=(8, 6))
   plt.title('Number of Relevant Tweets by Month')
   plt.savefig('img/relevantTweetsByMonth.png')

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
   numTopWords = 30
   wordlist = FreqDist(stems)
   word_features = [w for (w, c) in wordlist.most_common(numTopWords)]
   print(f'Top {numTopWords} Words: ', ", ".join(word_features))
   topFreq = FreqDist(word_features)
   topFreq.plot(title='Relevant Tweet Word Frequency', show=False)
   plt.plot(kind='bar')
   plt.savefig('img/relevantWordFrequency.png')
   wordcloud = WordCloud(
         width=800, 
         height=600,
         background_color='white',
         min_font_size=10
      ).generate_from_frequencies(dict(wordlist.most_common(200)))
   plt.figure(figsize=(8, 6), facecolor=None)
   plt.imshow(wordcloud)
   plt.axis('off')
   plt.tight_layout(pad=0)
   plt.savefig('img/relevantWordCloud.png')