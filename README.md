# Sentimental_Analysis
Sentiment analysis is the automated process of identifying and classifying subjective information in text data. This might be an opinion, a judgment, or a feeling about a particular topic or product feature. In this sentiment analysis, the most common type is ‘polarity detection’, which involves classifying statements as positive, negative, or neutral. It is a tool that automatically monitors emotions in conversations on social media platforms.

Pre-requisites
--------------

- Python
- Machine Learing

## STEPS

- Read the Data frame: it means importing the already classified data frame to train and test the 
model.

- Data Analysis: it means to find majority customers are positive or negative and finding the most 
used keywords.

- Classifying Tweets: it means we will classify reviews into “positive” and “negative,” so we can 
use this as training data for our sentiment classification model.

- Building the Model: we can build the sentiment analysis model. This model will take reviews in 
as input. It will then come up with a prediction on whether the review is positive or negative. For 
this we will split the data frame into train and test set of 80% and 20% respectively.

- Next, we will use a count vectorizer from the Scikit-learn library. This will transform the text in 
our data frame into a bag of words model, which will contain a sparse matrix of integers. The 
number of occurrences of each word will be counted and printed.

- Then we will train the model. Here we used the 4 methods SVM, Pipeline, Naive Bayes, and 
Multinomial NB to train and predict the output for the test set.

- Now we will find the accuracy of the model and can use the model to classify the given text to 
positive or negative



## Authors

- [@hemu33662](https://github.com/hemu33662)
