# Spooky Natural Language Processing Project


## The Problem

Natural Language Processing has become an important area of study within the last decade. The applications of the field are numerous, ranging from sentiment analysis to text classification to machine translation. Technological improvements and increase in accessibility to deep learning models has further unlocked the power of NLP.

My project is based on the “Spooky Author Identification” prediction competition outlined on the Kaggle website. The objective is to classify sentences by author, where each sentence is written by one of Edgar Allen Poe, HP Lovecraft, or Mary Shelley. I will develop a number of different models and identify which one performs best in this multinomial classification problem. The first step of developing this text classification model involves converting each sentence into a vector. There are a number of potential approaches here that include simpler vectorizers that implement a bag-of-words method as well as more complicated approaches such as deep learning embedding layers. The second step of the model includes using the vectorized text data to train different machine learning and deep learning models. 

Once I have identified the most effective model when it comes to classifying text, I will build an interactive python notebook that allows the user to input their own sentence and return information that indicates what author’s writing the input text resembles the most. 

## The Data

The data I will be using is part of a robust dataset provided by Kaggle for a 2018 competition. Each observation within the dataset is a sentence taken from a larger piece of text written by one of the three authors using CoreNLP's MaxEnt sentence tokenizer. The “target” column lists a three character acronym that represents the author the sentence was written by.

The raw dataset provided by Kaggle did not involve any missing data, so the main challenge involved transforming the text data into vectors that could be used to train a model. One hurdle that arose during the transformation step was the fact that the sentences were of different lengths (in terms of word count). This was resolved by adding padding to the vector after using the bag-of-words approach to transform the original text. Having a standard vector length for each sentence was necessary to successfully train the model. This was the extent of the data-wrangling done before training.

## Data Exploration

I did some initial exploration to get a better feel for the data. Despite the simplicity of the data there is still interesting information to be gleaned, especially when it comes to making use of visualization techniques. First, I generated wordclouds for each author that indicated what words were used the most frequently.  I combined the sentences for each individual author and filtered out words containing 4 characters or fewer so that the wordclouds didn’t include common pronouns, conjunctions, and other linking words. 

## Analysis

It is good to have a baseline level of accuracy to gauge the effectiveness of the models we develop. If there were an equal number of sentences for each author then a random guess approach would lead to an expected accuracy of ⅓. Selecting one of the authors for every sentence would produce an equivalent result. However, the sentences are not evenly distributed as seen below.
One naive method would just be to pick Edgar Allen Poe for each sentence, which would lead to a 40.3% accuracy. Additionally, we could randomly select an author based on the distribution (i.e. select Edgar Allen Poe 40.3% of the time, HPL 28.7% of the time, and MWS 30.8% of the time). This would produce an expected accuracy of 43%.

Once the vectorized text was split into training and test sets, I used some of the models offered in the sci-kit learn package to fit the model using the training set and predict on both the training and the test sets. 

The first model I trained was a simple multinomial logistic regression model, the model produced a training accuracy score of .984 and a test accuracy score of .812 suggesting that the model had an overfitting issue. Nonetheless the .812 percent accuracy was much higher than the baseline we discussed earlier at .432. To try to correct for this overfitting issue I constructed a ridge linear classifier, however this model produced both a lower training and test accuracy [.933, .807]. 

The second model I looked at was a multinomial naive bayes model, also part of the sci-kit learn package. The model produced a training accuracy score of .916 and a test accuracy score of .833. This test accuracy was even higher than the logistic regression model and since the difference between the training and test accuracy was smaller it indicated lower levers of overfitting. 

I constructed a decision tree classifier as well as a random forest classifier to look at some alternatives to move traditional models. Both models produced training accuracies close to one hundred percent, however they produced extremely poor testing accuracy when compared to the other models discussed above. The random forest classifier produced a testing accuracy of .613 while the decision tree produced a testing accuracy of .548. 

I also designed a number of different deep learning models to classify the sentences. The main structure I used was a sequential model provided by the keras framework. Different types of layers were used for each model. The first model I built involved an embedding layer that is typically used for NLP problems as it helps vectorize text effectively. The embedding layer is typically followed by a ‘Flatten’ layer that reduces the dimensionality of the output. I then added an additional dense layer, and finally an output layer with the softmax activation method. This simple deep learning model was trained for 3 epochs and produced a training accuracy of .941 and a testing accuracy of .785. While the training accuracy is higher than previous models discussed, the testing accuracy is a few percentage points lower. It appears that the deep learning model produces overfitting 

The second deep learning model I built involved the embedding layer, as well as the combination of a convolutional layer plus a max-pooling layer. The convolutional layer adds another dimension to each tensor and helps with the classification of sequential data, thus is useful when applied to text classification. After the convolutional layer is applied, the max-pooling layer is added to reduce the added dimensionality. This deep learning model produced a training accuracy of .943 and a testing accuracy of .798. Both accuracy values were higher than before suggesting that this new model is superior in classifying the text. 

I also swapped out the convolutional and max pooling layer for a Long Term Short Term Memory layer that utilizes memory to make more accurate predictions. This deep learning model produced a training accuracy of .882 and a testing accuracy of .823. Though this model is more accurate than the other deep learning model’s we’ve looked at so far, its testing accuracy still falls short of the naive bayes classifier discussed earlier. 












Google Doc Write Up: https://docs.google.com/document/d/1Y0oWUFiq08Hk1oVX7QTiDjnNFAaaITBmE9N4S8ltsaY/edit?usp=sharing
