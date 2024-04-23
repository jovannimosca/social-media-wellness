import pandas as pd
from textblob import TextBlob

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('data/bigboy.csv')

# Define the keywords you want to analyze for sentiment
keywords = ['yoga', 'fresh air', 'medication[a-z]*', 
            'vitamin[a-z]*', 'holistic[a-z]*', 'SSRIs', 
            'treatment[a-z]*', 'therapy', 'self help', 
            'self-help', 'cortisol', 'nature', 'natural', 
            'social[a-z]*', 'creativ[a-z]*', 'social media', 
            'news', 'alcohol', 'substance abuse', 'Lexapro', 
            'Prozac', 'Cipralex']

# Function to calculate sentiment polarity using TextBlob
def calculate_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Initialize a dictionary to store the count of positive sentiment for each keyword
positive_sentiment_counts = {keyword: 0 for keyword in keywords}

# Iterate over each keyword
for keyword in keywords:
    # Filter rows containing the current keyword
    filtered_df = df[df['full_text'].str.contains(keyword, case=False)].copy()
    
    # Apply sentiment analysis to filtered rows
    filtered_df.loc[:, 'sentiment_score'] = filtered_df['full_text'].apply(calculate_sentiment)

    total_tweets = len(filtered_df)
    
    # Count the number of positive sentiment occurrences
    positive_count = (filtered_df['sentiment_score'] > 0).sum()
    
    # Update the count in the dictionary
    positive_sentiment_counts[keyword] = [positive_count, total_tweets]
    
    # Print the keyword and the count of positive sentiment occurrences
    # print(f"Keyword: {keyword}, Positive Sentiment Count: {positive_count}")

# Print the dictionary containing positive sentiment counts for each keyword
print("Positive Sentiment Counts Compared to Total Count for Each Keyword:")
print(positive_sentiment_counts)