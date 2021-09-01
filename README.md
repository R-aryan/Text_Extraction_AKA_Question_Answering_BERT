# Text Extraction AKA Question Answering Using BERT

Performing Text Extraction also known as Question-Answering using BERT,and serving it Via REST API.

- End to End NLP  Text Extraction Probelm.
- The Kaggle dataset can be found Here [Click Here](https://www.kaggle.com/c/tweet-sentiment-extraction/data)

####  Steps to run the project [Click Here](https://github.com/R-aryan/Text_Extraction_AKA_Question_Answering_BERT/blob/main/backend/services/text_extraction/README.md)

### Dataset Description

### What should I expect the data format to be?
Each row contains the text of a tweet and a sentiment label. In the training set you are provided with a word or phrase drawn from the tweet (selected_text) that encapsulates the provided sentiment.

Make sure, when parsing the CSV, to remove the beginning / ending quotes from the text field, to ensure that you don't include them in your training.

### What am I predicting?

You're attempting to predict the word or phrase from the tweet that exemplifies the provided sentiment. The word or phrase should include all characters within that span (i.e. including commas, spaces, etc.)

### Example 
Let's say we have a sentence 
- **I am feeling good** and the sentiment for this sentence is **positive**
- The word/phrase which exemplifies the provided sentiment in this case is **good.**

### Goal
- So, the goal is to create a model which performs the above mentioned task.

### Columns
- textID - unique ID for each piece of text
- text - the text of the tweet
- sentiment - the general sentiment of the tweet
- selected_text - [train only] the text that supports the tweet's sentiment


 
