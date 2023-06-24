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
## Benefits

1. Understanding Customer Feelings: Sentiment analysis helps businesses understand how customers feel about their products, services, or brand. It identifies whether opinions are positive, negative, or neutral, providing valuable insights into customer satisfaction and preferences.

2. Improved Decision Making: By analyzing sentiments in customer feedback and social media conversations, businesses can make data-driven decisions to enhance their offerings, address customer concerns, and improve overall customer experience.

3. Real-Time Monitoring: Sentiment analysis enables real-time monitoring of customer sentiments. This helps businesses stay updated on customer opinions and promptly respond to any issues or emerging trends.

4. Reputation Management: Monitoring sentiment allows businesses to manage their brand reputation effectively. By addressing negative sentiments and resolving customer issues, companies can maintain a positive image and strengthen customer relationships.

5. Competitive Analysis: Sentiment analysis can be used to analyze sentiments related to competitors' products or services. This helps businesses gain insights into their strengths, weaknesses, and customer perceptions, enabling them to differentiate their offerings and develop effective marketing strategies.

6. Crisis Detection and Response: Sentiment analysis aids in detecting and managing potential crises by monitoring public sentiment. It allows businesses to identify and address negative sentiments quickly, minimizing the impact on their brand and reputation.

7. Social Listening: Sentiment analysis facilitates social listening, helping businesses track and evaluate customer responses to specific campaigns, events, or promotions. It enables the assessment of campaign success and provides valuable feedback for future improvements.

In simple terms, sentiment analysis helps businesses understand customer feelings, make better decisions, monitor sentiments in real time, manage their reputation, analyze competition, detect and respond to crises, and listen to customer feedback for improved marketing efforts.



## Authors

- [@hemu33662](https://github.com/hemu33662)
