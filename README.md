dshk1-loan-default
==================

Workshop on Kaggle's Loan Default dataset for General Assembly Hong Kong Data Science course

Competition website: http://www.kaggle.com/c/loan-default-prediction

Download the data: https://drive.google.com/file/d/0B7tb9lqp600SNE0zeFUwNE9NWlk/edit?usp=sharing
Sample submission file: https://drive.google.com/file/d/0B7tb9lqp600Sd1EzR01pQmttT0U/edit?usp=sharing

My contact: hgcrpd@gmail.com

Here's the link to my actual code: https://github.com/hxu/kaggle-loan-default

Pseudocode for overall pipeline:
================================

Get the training data X, and known values y
Figure out what model you want to run:
   - Selecting features
   - Cross validation - to find the best features
   - Grid search - to get the best parameters for our model

Get the training data X, and known values y
Set up the pipeline:
   - Extract the features we already selected
   - Cleaning up the data:
     - Fill NAs
     - Remove objects, unwanted columns
Fit the default model on X and y

Select the rows from Training data X and known values y that are predicted default
Set up the loss given default pipeline:
   - Extract the features
   - Clean up the data
Fit the loss (random forest) model on defaulted Xs and ys

Get the test data, test_X
Run it through default pipeline:
   - Extract features
   - Clean data in exact same way as the train data
Predict defaults = [0, 1, 0, 0, 0, 1] ~ 100,000

Select the rows from the test data that we think default
Run it through the loss pipeline
    - Extract the features
    - clean the data
Predict loss = [2, 48, 100,... 0]  ~ 10,000

Combine the predictions:
  If predicted default ( ==1) = set to loss
Final prediction = [0, 0, 0, 2, 0, 0, 48, 0, ..., 10] ~ 100,000

Write to file
Submit
