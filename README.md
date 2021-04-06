# Disaster Response Pipeline Project

This project processes message text received by aid organizations in disasters, and categorizes them across 36 key categories.  Primary categorizations are related/unrelated to a disaster, whether the message requests assistance, and whether the message is a direct report.  The full categorization includes the type of disaster, and the type of assistance requested in the message.

## Project Discussion

One of the challenges presented by this task is how to evaluate the performance of the model.  Accuracy is not necessarily the best metric here, as the data is unbalanced.  High accuracy scores can be achieved by a model which categorizes all messages as not in any of the categories.  So precision and recall are more appropriate.  Of the two, recall is the more important, since the impact of filtering out a request for aid is large compared to the impact of forwarding on a possibly-unrelated message.  However, precision cannot be discarded completely - the basis of the problem is that aid organizations receive a flood of messages and have trouble determining which messages are relevant to their response.  A perfect recall score could be achieved by simply forwarding on all messages, but this would accomplish nothing.  As a result, the machine learning models are evaluated on the basis of their f2 score, which gives slightly more weight to recall than precision.

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