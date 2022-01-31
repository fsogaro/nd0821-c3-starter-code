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

-----
## Learning's: dvc, github actions, heroku

##### folder structure
- make sure to deploy from root of repo Procfile, Aptfile and main.py should 
  be in the same level as the .github & .dvc folders (if there is a way to 
  deploy a sub folder to heroku I did not find it)
  

- once the dvc remote on aws is set up, you have an IAM role with s3 access 
  rights, create an secret key access for this role and use these secrets 
  for both github actions and heroku model
  
###### to make sure dvc works on heroku:
-  run `heroku buildpacks:add --index 1 heroku-community/apt` in the cd of 
  the project with heroku CLI
- have the Aptfile with the link to the dvc version
- have set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY as Config vars in heroku

###### to make sure dvc works on github actions:
- see .github/workflows/python-app.yml but ultimatly you need
- set the access keys and secret on the action, 
- setup and pull dvc

###### a post method requires something to post!
1. from local script see nd0821-c3-starter-code/call_post_api.py
2. from interactive, lauch the app, then navigate to "docs" e.g.
   https://sheltered-shelf-20033.herokuapp.com/docs and use the interactive post method
