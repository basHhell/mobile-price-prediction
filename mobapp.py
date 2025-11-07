
import joblib
from flask import Flask
import pandas as pd

app = Flask(__name__)

# Load trained model
loaded_model = joblib.load("log_reg.pkl")

features = [
    'battery_power','blue','clock_speed','dual_sim','fc','four_g',
    'int_memory','m_deep','mobile_wt','n_cores','pc','px_height','px_weight',
    'ram','sc_h','sm_w','talk_time','three_g','touch_screen','wifi'
]

@app.route('/')
def home():
    return "Welcome to the Mobile Price Prediction App"


@app.route('/predict', methods=['GET'])
def predict():
    input_data = {
        'battery_power': [0],
        'blue': [0],
        'clock_speed': [2],
        'dual_sim': [0],
        'fc': [0],
        'four_g': [0],
        'int_memory': [0],
        'm_deep': [0.2],
        'mobile_wt': [0],
        'n_cores': [0],
        'pc': [0],
        'px_height': [0],
        'px_weight': [0],
        'ram': [00],
        'sc_h': [0],
        'sm_w': [0],
        'talk_time': [0],
        'three_g': [0],
        'touch_screen': [0],
        'wifi': [1]
    }

    df = pd.DataFrame(input_data)
    prediction = loaded_model.predict(df)[0]
    return f"Predicted Price Range: {prediction}"


if __name__ == '__main__':
    app.run(debug=True,port=5001)



