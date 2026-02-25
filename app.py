import streamlit as st
import numpy as np
import pickle

# ---------------- LOAD MODELS ----------------
ln = pickle.load(open("linear_model.sav", "rb"))
sc = pickle.load(open("sc_model.sav", "rb"))

le1 = pickle.load(open("le1_model.sav", "rb"))   # transmission
le2 = pickle.load(open("le2_model.sav", "rb"))   # owner_type
ohe1 = pickle.load(open("ohe1_model.sav", "rb")) # brand

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Car Price Prediction", layout="centered")
st.title("🚗 Car Price Prediction")

# # ---------------- BACKGROUND IMAGE ----------------
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_img = get_base64("lights_car_dark_128635_1350x2400.jpg")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- USER INPUTS ----------------
engine_cc = st.number_input("Engine CC", min_value=500.0, max_value=6000.0)
mileage_kmpl = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=40.0)
horsepower = st.number_input("Horsepower", min_value=40.0, max_value=600.0)
km_driven = st.number_input("Kilometers Driven", min_value=0.0, max_value=500000.0)
maintenance_cost = st.number_input("Maintenance Cost", min_value=1000.0, max_value=50000.0)
accident_count = st.number_input("Accident Count", min_value=0, max_value=10, step=1)
warranty_left_years = st.number_input("Warranty Left (Years)", min_value=0.0, max_value=10.0)

fuel_type = st.selectbox(
    "Fuel Type", ["Hybrid", "Diesel", "Electric", "Petrol"]
)

transmission = st.selectbox(
    "Transmission", ["Manual", "Automatic"]
)

owner_type = st.selectbox(
    "Owner Type", ["First", "Second", "Third"]
)

brand = st.selectbox(
    "Brand", ["Audi", "BMW", "Ford", "Honda", "Hyundai", "Kia", "Toyota"]
)

# ---------------- ENCODING ----------------

# Fuel type (manual mapping – same as training)
fuel_map = {
    "Hybrid": 1,
    "Diesel": 2,
    "Electric": 3,
    "Petrol": 4
}
fuel_type_enc = fuel_map[fuel_type]

# Transmission & Owner type (LabelEncoder)
transmission_enc = le1.transform([transmission])[0]
owner_type_enc = le2.transform([owner_type])[0]

# Brand (OneHotEncoder)
brand_ohe = ohe1.transform([[brand]])  # numpy array

# ---------------- FINAL INPUT (ORDER MATTERS) ----------------
numerical_part = np.array([[
    fuel_type_enc,
    transmission_enc,
    owner_type_enc,
    engine_cc,
    mileage_kmpl,
    horsepower,
    km_driven,
    maintenance_cost,
    accident_count,
    warranty_left_years
]])

final_input = np.hstack((numerical_part, brand_ohe))

# Scale (IMPORTANT)
final_scaled = sc.transform(final_input)

# ---------------- PREDICTION ----------------
if st.button("Predict Price"):
    prediction = ln.predict(final_scaled)
    st.success(f"💰 Predicted Price: ₹ {prediction[0]:,.2f}")
