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













Google Doc Write Up: https://docs.google.com/document/d/1Y0oWUFiq08Hk1oVX7QTiDjnNFAaaITBmE9N4S8ltsaY/edit?usp=sharing
