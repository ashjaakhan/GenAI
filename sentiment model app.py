import joblib
import os



model_path="./model.pkl"


#Take input from user
input_text=input("Enter the text:")
cleaned_text=input_text.strip()
    
if os.path.exists(model_path):
    model =joblib.load(model_path)
    prediction_arr=model.predict([cleaned_text])
    prediction=prediction_arr[0]
    print(f"Prediction:{prediction}")
else:
    print("Model not found")

### app.py has noting to do with user_interface.py as app waalw se hum model cmd pr chala sakte hain
### aur user_interface se humne ek UI create krke seperately integrate kia hai humer model ko UI mai


        