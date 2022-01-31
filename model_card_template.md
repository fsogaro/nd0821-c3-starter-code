# Model Card: predicting income above or below $50k given inputs demographics

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
simple classification model to predict income above or below 50k, assignment 
3 of Udacity MLE nanodegree. The focus is less on model development but 
Ci/CD integration

## Intended Use
predictions available via post method to https://sheltered-shelf-20033.
herokuapp.com/predict

## Training Data
census data: https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
cross validation, 5 fold
- cv score mean:0.853
- cv score std:0.003

## Metrics
- precision: 0.738
- recall: 0.632
- fbeta: 0.681

## Ethical Considerations
bias factors such as sex, race are part of the dataset and performance 
slicing shows concerns over ceratain features

## Caveats and Recommendations
model is for education on MLE deplyment purposes only
