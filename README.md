# Disaster Response Pipeline Project

This project processes message text received by aid organizations in disasters, and categorizes them across 36 key categories.  Primary categorizations are related/unrelated to a disaster, whether the message requests assistance, and whether the message is a direct report.  The full categorization includes the type of disaster, and the type of assistance requested in the message.

## Dependencies
This project depends on the `pandas`, `numpy`, `scikit-learn`, `nltk` and `sqlalchemy` python libraries.  

Additional packages are used, but all within base python.

## Installation
From a new directory, using `conda`:  

```
conda create -n disaster_response
conda activate disaster_response
conda install numpy pandas scikit-learn nltk sqlalchemy
git clone https://github.com/chapman-mcd/Udacity_Disaster_Response
```

Elevated user privileges may be required the first time running train_classifier.py so that nltk can download the required supporting files to the appropriate directory.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Included Files

- /App/
	run.py: starts the webapp
    go.html, master.html: web source
- /data/process_data.py: processes message training data for consumption by the machine learning model
- /models/train_classifier.py: uses the processed data to train a message classifier

Training data not included as it is likely proprietary.

## Acknowledgements

- [FigureEight](https://appen.com/) (now acquired by Appen), for providing the training data
- [Udacity](http://www.udacity.com)

Part of a Data Scientist Nanodegree with Udacity.