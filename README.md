# xG_Football
Predicting xG using Bayesian Networks

This project is about using Probabilistic Graphical Models, like Bayesian Networks, to predict Expected Goals. I have also implemented a Gradient Boosting Classifier as a base model. Both the models use data from Kaggle's "Football-events" (https://www.kaggle.com/secareanualin/football-events) to train and subsequently are tested on real-life Football shots.
The contents of this repository include the codes for implementation of both the models along with the data used.
Both codes are written in Python.
GradientBoosting.py contains data preprocessing on raw data and subsequent GB model implementation. It uses scikit learn to implement and infer from the model.
BayesianNetworkModel.py contains a similar pre processing to get refined data for model training. The code uses the pgmpy package of Python for model implementation and inference (using variable elimination). 
The data consists of the events.csv file containing the raw data, dictionary.txt containing a guide to the data and ginf.csv which contains metadata and market odds about each game. 
The codes give output for the test cases (real matches shots) as xG for the particular shot. 
