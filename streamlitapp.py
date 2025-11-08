
import streamlit as st
import pandas as pd
import joblib

# Load both model and scaler
model = joblib.load("log_reg.pkl")
scaler = joblib.load("scaler.pkl")

features = [
    'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
    'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
    'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi'
]

st.title("📱 Mobile Price Prediction App")
st.write("Predict the price range of a mobile based on its specifications")

# Collect user inputs
input_data = {
    'battery_power': st.number_input("Battery Power (mAh)", min_value=500, max_value=2000, value=1000, key="bp"),
    'blue': st.selectbox("Bluetooth", [0,1], key="blue"),
    'clock_speed': st.number_input("Clock Speed (GHz)", min_value=0.5, max_value=3.0, value=1.5, key="clk"),
    'dual_sim': st.selectbox("Dual Sim", [0,1], key="dual"),
    'fc': st.number_input("Front Camera (MP)", min_value=0, max_value=20, value=5, key="fc"),
    'four_g': st.selectbox("4G Supported", [0,1], key="four"),
    'int_memory': st.number_input("Internal Memory (GB)", min_value=2, max_value=512, value=64, key="int"),
    'm_dep': st.number_input("Mobile Depth (cm)", min_value=0.1, max_value=1.0, value=0.5, key="dep"),
    'mobile_wt': st.number_input("Mobile Weight (g)", min_value=80, max_value=250, value=150, key="wt"),
    'n_cores': st.number_input("No. of Cores", min_value=1, max_value=8, value=4, key="cores"),
    'pc': st.number_input("Primary Camera (MP)", min_value=0, max_value=20, value=10, key="pc"),
    'px_height': st.number_input("Pixel Height", min_value=0, max_value=2000, value=900, key="pxh"),
    'px_width': st.number_input("Pixel Width", min_value=0, max_value=2000, value=1500, key="pxw"),
    'ram': st.number_input("RAM (MB)", min_value=256, max_value=8192, value=4096, key="ram"),
    'sc_h': st.number_input("Screen Height (cm)", min_value=5, max_value=20, value=12, key="sch"),
    'sc_w': st.number_input("Screen Width (cm)", min_value=2, max_value=10, value=6, key="scw"),
    'talk_time': st.number_input("Talk Time (hrs)", min_value=2, max_value=20, value=10, key="talk"),
    'three_g': st.selectbox("3G Supported", [0,1], key="three"),
    'touch_screen': st.selectbox("Touch Screen", [0,1], key="touch"),
    'wifi': st.selectbox("Wi-Fi Supported", [0,1], key="wifi")
}

if st.button("🔮 Predict"):
    df = pd.DataFrame([input_data])

    # ✅ Apply the same scaling
    df_scaled = scaler.transform(df)

    # ✅ Predict using scaled data
    prediction = model.predict(df_scaled)[0]

    labels = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
    st.success(f"📊 Predicted Price Range: **{labels[prediction]}**")

