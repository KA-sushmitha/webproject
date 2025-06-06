import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("ckd_model.pkl", "rb"))

st.title("🌡️ Chronic Kidney Disease (CKD) Prediction")
st.write("Fill in the following details to predict CKD risk:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=100)
bp = st.number_input("Blood Pressure (bp)", min_value=0, max_value=200)
sg = st.selectbox("Specific Gravity (sg)", [1.005, 1.010, 1.015, 1.020, 1.025])
al = st.selectbox("Albumin (al)", [0, 1, 2, 3, 4, 5])
su = st.selectbox("Sugar (su)", [0, 1, 2, 3, 4, 5])
rbc = st.selectbox("Red Blood Cells (rbc)", ["normal", "abnormal"])
pc = st.selectbox("Pus Cell (pc)", ["normal", "abnormal"])
pcc = st.selectbox("Pus Cell Clumps (pcc)", ["present", "notpresent"])
ba = st.selectbox("Bacteria (ba)", ["present", "notpresent"])
bgr = st.number_input("Blood Glucose Random (bgr)", min_value=20, max_value=500)
bu = st.number_input("Blood Urea (bu)", min_value=1, max_value=300)
sc = st.number_input("Serum Creatinine (sc)", min_value=0.1, max_value=50.0)
sod = st.number_input("Sodium (sod)", min_value=100, max_value=200)
pot = st.number_input("Potassium (pot)", min_value=2.0, max_value=10.0)
hemo = st.number_input("Hemoglobin (hemo)", min_value=3.0, max_value=20.0)
pcv = st.number_input("Packed Cell Volume (pcv)", min_value=10, max_value=60)
wbcc = st.number_input("White Blood Cell Count (wbcc)", min_value=1000, max_value=25000)
rbcc = st.number_input("Red Blood Cell Count (rbcc)", min_value=2.0, max_value=7.0)
htn = st.selectbox("Hypertension (htn)", ["yes", "no"])
dm = st.selectbox("Diabetes Mellitus (dm)", ["yes", "no"])
cad = st.selectbox("Coronary Artery Disease (cad)", ["yes", "no"])
appet = st.selectbox("Appetite (appet)", ["good", "poor"])
pe = st.selectbox("Pedal Edema (pe)", ["yes", "no"])
ane = st.selectbox("Anemia (ane)", ["yes", "no"])

# Encode text fields
binary_map = {"yes": 1, "no": 0, "present": 1, "notpresent": 0, "normal": 0, "abnormal": 1, "good": 0, "poor": 1}
encoded_input = [
    age, bp, sg, al, su,
    binary_map[rbc], binary_map[pc], binary_map[pcc], binary_map[ba],
    bgr, bu, sc, sod, pot,
    hemo, pcv, wbcc, rbcc,
    binary_map[htn], binary_map[dm], binary_map[cad],
    binary_map[appet], binary_map[pe], binary_map[ane]
]

if st.button("🔍 Predict"):
    input_data = np.array([encoded_input])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("❌ High Risk of Chronic Kidney Disease (CKD).🩺Consult a Doctor.")
    else:
        st.success("✅ Low Risk of Chronic Kidney Disease (CKD)")
