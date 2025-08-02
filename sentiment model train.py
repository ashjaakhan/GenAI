import pandas as pd
import numpy as np
import joblib

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

#Baye's theorem classification
from sklearn.naive_bayes import MultinomialNB

#Making model flow | ML pipeline | 
from sklearn.pipeline import make_pipeline

#Reading the CSV dataset
data = pd.read_csv(r"C:\Users\Mohd Ashjaa Khan\Downloads\sentiment_dataset_50k.csv",encoding="ISO-8859-1")

# Convert to DataFrame
df = pd.DataFrame(data)

#Encoding techniques for the classification
#label Encode
df["sentiment"]=df["sentiment"].astype("category")

# Dropping the Nan or null data in the dataset for being double sure
df =df.dropna(subset=["text","sentiment"])

# Dropping the Nan or null data in the text and sentiment labels as there are unequal number of rows in the dataset under text and sentiment columns
X=df["text"]
Y=df["sentiment"]

#split and test the data and train
X_train,X_test,Y_train,Y_test=train_test_split(df["text"],df["sentiment"],test_size=0.2) #80-20 rule (20% test data mai 80% accuracy)

#Make the pipeline
model=make_pipeline(CountVectorizer(),MultinomialNB())

#curvefitting
model.fit(X_train,Y_train)

#Training and Saving model
if joblib.dump(model,"model.pkl"):
    print("Model trained and saved")
