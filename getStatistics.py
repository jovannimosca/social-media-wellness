import pandas as pd
import matplotlib.pyplot as plt
from nltk import *
from nltk.corpus import stopwords
from wordcloud import WordCloud
from processData import filterHashtags
import string, re, spacy

def getMonthlyVolume(df:pd.DataFrame) -> pd.DataFrame:
   """Get the monthly tweet volume for a given data frame and return a new data frame with the month names and count of tweets.

   Parameters
   ----------
   df : pd.DataFrame
       The source dataframe to plot.
   
   Returns
   -------
   pd.DataFrame
       A transformed data fram containing a 'month' column with the month name and a 'count' column with the number of tweets.
   """
   # Let's see statistics by year:
   # It looks like the vast majority of data is from 2021, so let's limit to that:
   print(f'Generating monthly graph... ', end='')
   monthly = df[df['publicationtime'].dt.year == 2021].groupby(pd.Grouper(key='publicationtime', freq='ME'))
   # monthly = df.groupby(pd.Grouper(key='publicationtime', freq='ME'))
   monthly = monthly.size().reset_index(name='count')
   monthly['month'] = monthly['publicationtime'].dt.strftime('%B')
   print('Done')
   return monthly
   
   
def tokenizeCorpus(df:pd.DataFrame, unique:bool=False) -> list[str]:
   """Tokenize an entire corpus of tweets into its words.

   Parameters
   ----------
   df : pd.DataFrame
       The source dataframe with a 'full_text' column.
   unique : bool, optional
       Only return unique tokens, by default False

   Returns
   -------
   list[str]
       A list of tokens, all lowercase.
   """
   tt = TweetTokenizer(strip_handles=True)
   corpusTokens = []
   for tweet in df['full_text']:
      for token in tt.tokenize(tweet):
         token = token.lower().strip(' #@')
         if token not in stopwords.words('english') and token not in string.punctuation and not token.startswith('http') and token not in [r'`', '\u2019']:
            if not unique or (unique and token not in corpusTokens): corpusTokens.append(token)
   return corpusTokens

def generateWordcloud(freq_wordlist:dict[str, float], path:str) -> None:
   """Generate a word cloud given a set of words and frequencies.

   Parameters
   ----------
   freq_wordlist : dict[str, float]
       The frequency distribution of words, a mapping from word to frequency.
   path : str
       The location to save the generated word cloud.
   """
   wordcloud = WordCloud(
         width=800, 
         height=600,
         background_color='white',
         min_font_size=10
      ).generate_from_frequencies(freq_wordlist)
   plt.figure(figsize=(8, 6), facecolor=None)
   plt.imshow(wordcloud)
   plt.axis('off')
   plt.tight_layout(pad=0)
   plt.savefig(path)

if __name__ == '__main__':
   # Get processed data:
   print('Reading from CSV...', end=' ')
   ingested = pd.read_csv('processed/cleaned.csv', parse_dates=['publicationtime']).convert_dtypes()
   filtered = pd.read_csv('processed/relevant.csv', parse_dates=['publicationtime']).convert_dtypes()
   print('Done')
   print(f'Total number of tweets: {len(ingested)}')
   print(f'Number of filtered tweets: {len(filtered)} ({len(filtered)/len(ingested)*100:.2f}%)')
   
   nlp = spacy.load("en_core_web_sm")
   
   # Get volume of tweets to view trends:
   allMonthly = getMonthlyVolume(ingested)
   filteredMonthly = getMonthlyVolume(filtered)
   # Plot both on the same graph:
   plt.figure(figsize=(8, 6))
   plt.subplots_adjust(bottom=0.20)
   plt.plot(allMonthly['month'], allMonthly['count'], label='All')
   plt.plot(filteredMonthly['month'], filteredMonthly['count'], label='Filtered')
   plt.title('Tweets by Month')
   plt.xlabel('Month')
   plt.ylabel('Occurrences')
   plt.legend()
   plt.xticks(rotation='vertical')
   plt.savefig('img/tweetsByMonth.png')
   plt.clf()
   # Plot the filtered only:
   filteredMonthly.plot(x='month', y='count', kind='line', figsize=(8, 6), title='Relevant Tweets by Month', xlabel='Month', ylabel='Occurrences', color='tab:orange')
   plt.savefig('img/relevantTweetsByMonth.png')


   # Tokenization:
   print('Tokenizing corpus... ', end='')
   corpusTokens = tokenizeCorpus(filtered)
   print('Done')
            
   # Stemming:
   print('Stemming corpus... ', end='')
   stemmer = PorterStemmer()
   stems = [stemmer.stem(token) for token in corpusTokens]
   print('Done')
   print(f'Stems: {stems[:20]}...')
   
   # POS Tagging:
   print('POS Tagging... ', end='')
   taggedWords = pos_tag(corpusTokens)
   corpusNouns = [w for (w, tag) in taggedWords if re.match(r'NN\d*', tag)]
   print('Done\n')
   print('Tagged: ', taggedWords[:20], '...\n')
   print('Nouns: ', corpusNouns[:20], '...\n')

   # Frequency:
   print('Calculating word frequency... ', end='')
   numTopWords = 30
   wordlist = FreqDist(corpusTokens)
   wordlistStems = FreqDist(stems)
   wordlistNouns = FreqDist(corpusNouns)
   print('Done')
   print(f'Top {numTopWords} Words: ', ", ".join([f'{w} ({c})' for (w, c) in wordlist.most_common(20)]), '\n')
   print(f'Top {numTopWords} Word Stems: ', ", ".join([f'{w} ({c})' for (w, c) in wordlistStems.most_common(20)]), '\n')
   print(f'Top {numTopWords} Nouns: ', ", ".join([f'{w} ({c})' for (w, c) in wordlistNouns.most_common(20)]), '\n')
   # Plot normal keywords:
   plt.figure(figsize=(8, 6))
   plt.subplots_adjust(bottom=0.30)
   plt.bar([w for (w, _) in wordlist.most_common(numTopWords)], [c for (_, c) in wordlist.most_common(numTopWords)])
   plt.title('Relevant Tweet Word Frequency')
   plt.xticks(rotation='vertical')
   plt.xlabel('Word')
   plt.ylabel('Occurrences')
   plt.savefig('img/relevantWordFrequency.png')
   plt.clf()
   # Plot stemmed:
   plt.bar([w for (w, _) in wordlistStems.most_common(numTopWords)], [c for (_, c) in wordlistStems.most_common(numTopWords)], color='tab:orange')
   plt.title('Relevant Tweet Word Frequency (Stemmed)')
   plt.xticks(rotation='vertical')
   plt.xlabel('Word')
   plt.ylabel('Occurrences')
   plt.savefig('img/relevantStemFrequency.png')
   plt.clf()
   # Plot nouns:
   plt.bar([w for (w, _) in wordlistNouns.most_common(numTopWords)], [c for (_, c) in wordlistNouns.most_common(numTopWords)], color='tab:green')
   plt.title('Relevant Tweet Word Frequency (Nouns)')
   plt.xticks(rotation='vertical')
   plt.xlabel('Word')
   plt.ylabel('Occurrences')
   plt.savefig('img/relevantNounFrequency.png')
   plt.clf()
   
   
   # Get stats from hashtags:
   print('Calculating most frequent hashtags... ', end='')
   numTopHashtags = 10
   relevantHashtags = []
   for tagset in filtered['hashtags']:
      hashtags = tagset.split(',')
      hashtags = [tag.lower().strip() for tag in hashtags]
      relevantHashtags.extend(hashtags)
   hashtagFreq = FreqDist(relevantHashtags)
   topHashtags = [w for (w, _) in hashtagFreq.most_common(numTopHashtags)]
   topTagDF = filterHashtags(filtered, topHashtags)
   # Save top hashtag tweets to new CSV and save top hashtags to text file:
   topTagDF.to_csv('processed/topTagTweets.csv')
   with open('processed/topTags.txt', 'w') as f:
      f.write(",".join(topHashtags))
   print('Done')
   print('Top Hashtags: ', topHashtags[:20], '\n')
   
   # NER:
   # doc = nlp(" ".join(corpusTokens)[:99999])
   # eventWords = [entity.text for entity in doc.ents if entity.label_ == "EVENT"]
   print('Beginning NER... ', end='')
   eventWords = []
   for tweet in filterHashtags(filtered, topHashtags)['full_text']:
      doc = nlp(tweet)
      eventWords.extend([entity.text for entity in doc.ents if entity.label_ == "EVENT"])
   eventFreq = FreqDist(eventWords)
   tt = TweetTokenizer(strip_handles=True)
   eventTokens = tt.tokenize(" ".join(eventWords))
   eventTFreq = FreqDist(eventTokens)
   print('Done\n')
   print(f'NER found some event-related tokens: {eventWords}')
   generateWordcloud(dict(eventTFreq.most_common(10000)), 'img/eventWordCloud.png')
   topEvents = pd.DataFrame(list(eventFreq.items()), columns=['word', 'frequency'])
   topEvents.to_csv('processed/topEvents.csv')
   
   # Generate wordclouds from frequencies:
   generateWordcloud(dict(wordlist.most_common(10000)), 'img/relevantWordCloud.png')
   generateWordcloud(dict(wordlistStems.most_common(10000)), 'img/relevantStemCloud.png')
   generateWordcloud(dict(wordlistNouns.most_common(10000)), 'img/relevantNounCloud.png')