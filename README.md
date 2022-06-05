# bda_project

## Speech Emotion Recognition
We have build a speech emotion recognition classifier

## What is speech emotion recognition (SER) ?

* Speech Emotion Recognition, abbreviated as SER, is the act of attempting to recognize human emotion and affective states from speech. This is capitalizing on the fact that voice often reflects underlying emotion through tone and pitch. This is also the phenomenon that animals like dogs and horses employ to be able to understand human emotion.

## Why we need it ?

1. Emotion recognition is the part of speech recognition which is gaining more popularity and need for it increases enormously. Although there are methods to recognize emotion using machine learning techniques, this project attempts to use deep learning to recognize the emotions from data.

2. SER(Speech Emotion Recognition) is used in call center for classifying calls according to emotions and can be used as the performance parameter for conversational analysis thus identifying the unsatisfied customer, customer satisfaction and so on.. for helping companies improving their services

3. It can also be used in-car board system based on information of the mental state of the driver can be provided to the system to initiate his/her safety preventing accidents to happen

## Datasets used in this project

* Crowd-sourced Emotional Mutimodal Actors Dataset (Crema-D)
* Ryerson Audio-Visual Database of Emotional Speech and Song (Ravdess)
* Surrey Audio-Visual Expressed Emotion (Savee)
* Toronto emotional speech set (Tess)

## Roles of team members

1. Preetish Patel - Combining all the four datasets into single dataframe, Data Visualisation and Exploration, Data augmentation, Feature extraction, Data Preparation
2. Abhinav Arora - Building a neural network model on preprocessed data, achieving the best accuracy of 60% on 8 classes after a series of experiments
3. Vatsal Savaliya & Abhinay Pandey - Deploying the saved model on web application using flask and creating a basic design of website using html

## How to run the project on your system ?

* Download the complete code, create a virtual environment the navigate to `bda_project/` folder run the command `pip install -r requirements.txt`.
* once the requirements are install run the command `python main.py`
