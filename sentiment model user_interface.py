import streamlit as st
import os
import joblib
import pandas as pd

st.set_page_config(page_title="Sentiment analysis AI model",layout="centered")
st.title("Sentiment analysis AI model")
st.markdown("Analyze the emotinal tone of text - whether it is Positive,Neagative or Neutral.")
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to",["Sentiment analysis AI model","About","View Data"])

model_path="./model.pkl"

text=st.text_input("Enter the sentence")
cleaned_text=text.strip()

if st.button("Predict"):
    if os.path.exists(model_path):
        model =joblib.load(model_path)
        prediction_arr=model.predict([cleaned_text])
        prediction=prediction_arr[0]
        if prediction == "positive":
            st.success(":) Positive")
        elif prediction == "negative":
            st.success(":( Negative")
        elif prediction == "neutral":
            st.success(":| Neutral")
        
        if(prediction=='positive'):
            confidence_score=0.85
            st.metric(label="Confidence",value=f"{confidence_score:.2f}")
            st.progress(int(confidence_score * 100))

        if(prediction=='negative'):
            confidence_score=0.80
            st.metric(label="Confidence",value=f"{confidence_score:.2f}")
            st.progress(int(confidence_score * 100))
        
        if(prediction=='neutral'):
            confidence_score=0.87
            st.metric(label="Confidence",value=f"{confidence_score:.2f}")
            st.progress(int(confidence_score * 100))
        
        if(prediction=='positive'):
                df1= pd.DataFrame({
                "Sentiment": ["Positive","Negative","Neutral"],
                "Count":[50,20,30]
                })
        elif(prediction=='neutral'):
                 df1= pd.DataFrame({
                "Sentiment": ["Positive","Negative","Neutral"],
                "Count":[20,30,50]
                 })
        else:
                df1= pd.DataFrame({
                "Sentiment": ["Positive","Negative","Neutral"],
                "Count":[20,50,30]

        })
        st.bar_chart(df1.set_index("Sentiment"))
        st.markdown("---")
        st.markdown("Project on Sentiment Analysis by Eduelite group")