from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging

app = Flask(__name__)

label_encoders = joblib.load("models/label_encoders.pkl")
clip_values = joblib.load("models/clip_values.pkl")
scaler = joblib.load("models/scaler.pkl")
rf_model = joblib.load("models/rf_model.pkl")
dt_model=joblib.load("models/dt_model.pkl")
knn=joblib.load("models/knn.pkl")


# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    
    date_str = request.form.get('date of reservation')
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    except:
        date_obj = datetime.strptime('2018-02-28', '%Y-%m-%d') 

    raw_data = {
        "number of adults": int(request.form.get('number of adults', 0)),
        "number of children": int(request.form.get('number of children', 0)),
        "number of weekend nights": int(request.form.get('number of weekend nights', 0)),
        "number of week nights": int(request.form.get('number of week nights', 0)),
        "type of meal": request.form.get('type of meal'),
        "car parking space": int(request.form.get('car parking space', 0)),
        "room type": request.form.get('room type'),
        "lead time": int(request.form.get('lead time', 0)),
        "market segment type": request.form.get('market segment type'),
        "repeated": int(request.form.get('repeated', 0)),
        "P-C": int(request.form.get('P-C', 0)),
        "P-not-C": int(request.form.get('P-not-C', 0)),
        "average price": float(request.form.get('average price', 0)),
        "special requests": int(request.form.get('special requests', 0))
    }
    data = pd.DataFrame([raw_data])

    data['day'] = date_obj.day
    data['Year'] = date_obj.year
    data['Month'] = date_obj.month
    data['Total_Guests'] = data['number of adults'] + data['number of children']
    data['Total_Nights'] = data['number of weekend nights'] + data['number of week nights']



     # ------------------- Clean and encode categorical columns -------------------
    categorical_cols = ["type of meal", "room type", "market segment type"]
    for col in categorical_cols:
        data[col] = data[col].astype(str).str.strip()  # remove spaces

        try:
            data[col] = label_encoders[col].transform(data[col])
        except ValueError as e:
            logging.warning(f"Encoding error in column '{col}': {e}")
            # fallback to 0 if unseen value
            data[col] = 0


    final_columns = [
    "number of adults","number of children","number of weekend nights",
    "number of week nights","type of meal","car parking space","room type",
    "lead time","market segment type","repeated","P-C","P-not-C","average price",
    "special requests","day","Year","Month","Total_Guests","Total_Nights"
    ]
            
    data = data[final_columns]

    data["lead time"] = data["lead time"].clip(upper=clip_values["lead_time_upper"])
    data["average price"] = data["average price"].clip(upper=clip_values["avg_price_upper"])

    data_scaled = scaler.transform(data.values)

    pred_rf = rf_model.predict(data_scaled)[0]
    pred_knn = knn.predict(data_scaled)[0]
    pred_dt = dt_model.predict(data_scaled)[0]

    final_prediction = np.bincount([pred_rf, pred_knn, pred_dt]).argmax()


    result = "Booking will be cancelled" if final_prediction == 1 else "Booking will not be cancelled"


    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)

    
    
    