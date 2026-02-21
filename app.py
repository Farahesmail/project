from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

app = Flask(__name__)

label_encoders = joblib.load("models/label_encoders.pkl")
clip_values = joblib.load("models/clip_values.pkl")
scaler = joblib.load("models/scaler.pkl")
rf_model = joblib.load("models/rf_model.pkl")
dt_model=joblib.load("models/dt_model.pkl")
knn=joblib.load("models/knn.pkl")

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
    data['Month_Name'] = date_obj.strftime('%B') # Month_Name اللي الموديل محتاجه
    data['Total_Guests'] = data['number of adults'] + data['number of children']
    data['Total_Nights'] = data['number of weekend nights'] + data['number of week nights']


    for col in label_encoders:
      
      val = request.form.get(col)
    
    # تنظيف القيمة: إزالة أي مسافات في الأول أو الآخر وتوحيد حالة الأحرف
    if isinstance(val, str):
        val = val.strip() # ده هيشيل أي مسافة زيادة أنتِ مش شايفاها
    
    try:
        # بنحول القيمة لـ List عشان الـ transform بيحتاج كده
        if col != "booking status":
         data[col] = label_encoders[col].transform([val])[0]
    except ValueError:
        # لو لسه مش لاقيها، اطبعي القيمة في الـ Terminal عشان نشوف الفرق
        print(f"Mismatch found! Encoder expects categories: {label_encoders[col].classes_}")
        print(f"But received value: '{val}'")
        # حل مؤقت عشان البرنامج يكمل:
        data[col] = 0


    #encoded_cols = ["type of meal", "market segment type", "room type", "Month_Name"]
    #for col in encoded_cols:
        # strip() بتشيل أي مسافات مخفية مسببة الـ ValueError
       ## val = str(data[col].iloc[0]).strip()
      #  data[col] = label_encoders[col].transform([val])

    final_columns = [
        "number of adults", "number of children", "number of weekend nights", 
        "number of week nights", "type of meal", "car parking space", 
        "room type", "lead time", "market segment type", "repeated", 
        "P-C", "P-not-C", "average price", "special requests", 
        "day", "Year", "Month", "Month_Name", "Total_Guests", "Total_Nights"
    ]
    # إعادة ترتيب DataFrame بناءً على القائمة أعلاه
    data = data[final_columns]
      # Encoding
    #for col in label_encoders:
       # if col != "booking status":
        #    data[col] = label_encoders[col].transform(data[col])

    data["lead time"] = data["lead time"].clip(upper=clip_values["lead_time_upper"])
    data["average price"] = data["average price"].clip(upper=clip_values["avg_price_upper"])

    data_scaled = scaler.transform(data)

    pred_rf = rf_model.predict(data_scaled)
    pred_knn = knn.predict(data_scaled)
    pred_dt = dt_model.predict(data_scaled)

    final_prediction = np.bincount([pred_rf, pred_knn, pred_dt]).argmax()


    result = "Booking will be cancelled" if final_prediction == 1 else "Booking will not be cancelled"


    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)