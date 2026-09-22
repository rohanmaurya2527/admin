#prac2
!pip install textblob
from textblob import TextBlob
text = " I am very sad today. But my friend got promotion which is somehow makes me happy."
# Create a TextBlob object
blob = TextBlob(text)
# Get the sentiment polarity (-1 to 1)
polarity = blob.sentiment.polarity
# Get the sentiment subjectivity (0 to 1)
subjectivity = blob.sentiment.subjectivity
print(f"Text: {text}")
print(f"Polarity: {polarity}")
print(f"Subjectivity: {subjectivity}")
if polarity > 0:
    print("Overall Sentiment: Positive")
elif polarity < 0:
    print("Overall Sentiment: Negative")
else:
    print("Overall Sentiment: Neutral")
