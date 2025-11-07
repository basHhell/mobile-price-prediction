import streamlit as st
import pandas as pd
import joblib

loaded_model=joblib.load("log_reg.pkl")

features = [
    'battery_power','blue','clock_speed','dual_sim','fc','four_g',
    'int_memory','m_deep','mobile_wt','n_cores','pc','px_height','px_weight',
    'ram','sc_h','sm_w','talk_time','three_g','touch_screen','wifi'
]

st.title("Mobile Prediction App")
st.write("Predict whether mobile price is Low,Medium,High,Very High")

input_data= {'battery_power': st.number_input("Battery Power", min_value=500, max_value=2000),
             'blue':st.selectbox("Bluetooth", [0,1]),
             'clock_speed':st.number_input("Clock Speed", min_value=0.5, max_value=3.0),
             'dual_sim':st.selectbox("Dual Sim",[0,1]),
             'fc':st.number_input("Front Camera",min_value=8,max_value=50),
             'four_g':st.selectbox("Four-G",[0,1]),
             'int_memory':st.number_input("Int Memory", min_value=60, max_value=500),
            'm_deep':st.number_input("Mobile Depth (cm)", min_value=0.1, max_value=1.0),
            'mobile_wt':st.number_input("Weight (grams)", min_value=80, max_value=250, value=150),
            'n_cores':st.number_input("Number of Cores", min_value=1, max_value=8, value=4),
            'pc':st.number_input("Primary Camera (MP)", min_value=0, max_value=20, value=10),
            'px_height':st.number_input("Pixel Resolution", min_value=0, max_value=2000, value=800),
            'px_weight':st.number_input("Pixel Resolution", min_value=0, max_value=2000, value=1000),
            'ram':st.number_input("RAM (MB)", min_value=256, max_value=8192, value=2048),
            'sc_h':st.number_input("Screen Height (cm)", min_value=5, max_value=20, value=10),
            'sm_w':st.number_input("Screen Width (cm)", min_value=2, max_value=10, value=5),
            'talk_time':st.number_input("Talk Time (hours)", min_value=2, max_value=20, value=10),
            'three_g':st.selectbox("3G Supported", [0, 1]),
            'touch_screen':st.selectbox("Touch Screen", [0, 1]),
            'wifi':st.selectbox("Wi-Fi Supported", [0, 1])
             }

df=pd.DataFrame([input_data])
if st.button("Predict"):
    prediction=loaded_model.predict(df)[0]
    st.success(prediction)

    labels = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
    st.success(f"📊 Predicted Mobile Price Range: **{labels[prediction]}**")
