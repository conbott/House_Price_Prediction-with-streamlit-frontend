import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
st.set_page_config(
    page_title="AI House Price Estimator",
    layout="wide"
)

st.markdown("""
    <style>
    .title-header { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.2rem; }
    .subtitle { color: #64748B; font-size: 1.05rem; margin-bottom: 1.8rem; }
    .result-card { background: #F8FAFC; border-radius: 12px; padding: 24px; border: 1px solid #E2E8F0; text-align: center; }
    .price-text { font-size: 2.6rem; font-weight: 800; color: #059669; }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = "saved_models/house_price_pipeline.pkl"
@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)
pipeline = load_pipeline()
STATES = [
    'Massachusetts', 'Connecticut', 'New Hampshire', 'Vermont','New Jersey', 'New York', 'Rhode Island', 'Maine', 'Pennsylvania', 'Delaware'
]

state = st.sidebar.selectbox("State / Location", STATES, index=0)
beds = st.sidebar.slider("Bedrooms", min_value=1, max_value=10, value=3, step=1)
baths = st.sidebar.slider("Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
house_size = st.sidebar.number_input("Living Area (sq ft)", min_value=300, max_value=12000, value=1850, step=50)
acre_lot = st.sidebar.number_input("Lot Size (Acres)", min_value=0.05, max_value=20.0, value=0.25, step=0.05)

if pipeline is None:
    st.error("Model not found! Ensure `saved_models/house_price_pipeline.pkl` exists.")
    st.stop()

col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.subheader("Selected Property Profile")
    m1, m2 = st.columns(2)
    m1.metric("Location", state)
    m2.metric("Rooms", f"{beds} Bed / {baths} Bath")
    
    m3, m4 = st.columns(2)
    m3.metric("Living Area", f"{house_size:,} sq ft")
    m4.metric("Lot Size", f"{acre_lot:.2f} Acres")
    
    st.divider()
    predict_btn = st.button("🔮 Calculate Market Valuation", type="primary", use_container_width=True)

with col2:
    st.subheader("Valuation Result")
    if predict_btn:
        input_data = pd.DataFrame([{
            'bed': beds,
            'bath': baths,
            'acre_lot': acre_lot,
            'house_size': house_size,
            'state': state
        }])
        
        predicted_price = pipeline.predict(input_data)[0]

        st.markdown(f"""
            <div class="result-card">
                <span style="font-size: 0.95rem; color: #64748B; font-weight: 600;">ESTIMATED MARKET VALUE</span>
                <div class="price-text">${predicted_price:,.0f}</div>
                <hr style="margin: 14px 0; border: none; border-top: 1px solid #E2E8F0;">
                <span style="font-size: 0.9rem; color: #475569;">
                    Estimated Range (±8%): <b>${predicted_price*0.92:,.0f} - ${predicted_price*1.08:,.0f}</b>
                </span>
            </div>
        """, unsafe_allow_html=True)
        st.success("Valuation generated successfully from the trained pipeline.")
    else:
        st.info("Set your property details in the sidebar and click **'Calculate Market Valuation'**.")
