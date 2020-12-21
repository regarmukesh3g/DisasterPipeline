# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#Installation)
2. [Project Motivation](#Motivation)
3. [File Descriptions](#descriptions)
4. [Instructions](#Instructions)
5. [Results](#Results)


### Installation <a name='Installation'>:
The requierement for this project is as following:- <br>
 1. Python 3.8 .
 2. pandas.
 3. flask.
 4. plotly.
 5. nltk.
 
 
### Project Motivation <a name='Motivation'>:
For this project I was interested in classifying messages sent during disaster into categories. One message can be
related to several entities, so I have trained a classifier to get multiple outputs and see what all categories a
message belongs to.
I have added a webapp as well so that user can make predictions from web UI.

### File Descriptions <a name='descriptions'>:
There are 3 folders in this project. And README file is there for help.
The directory webapp has all the webapp implementation. It is based on flask and standard structure is there for understanding.
The data directory contains the data used for this project and used for saving data(database tables) in the flow.
The models directory contains the files for training the model and classifier is saved into this as a pickle file.


### Instructions <a name='Instructions'>:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Results <a name='Results'>:
The model was trained for different categories with more than 90% accuracy for most of them.
The web UI is successfully able to display the categories a message belongs to.




