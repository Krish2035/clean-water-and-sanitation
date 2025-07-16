import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Water Access Predictor", layout="wide")

# Load dataset
data = pd.read_csv("expanded_country_water_access_data.csv")
countries = sorted(data['Country'].unique())

# --- Custom CSS ---
st.markdown("""
    <style>
        .big-font { font-size: 24px !important; font-weight: bold; color: #4A90E2; }
        .small-note { font-size: 13px; color: #666; }
        .highlight { background-color: #E6F0FF; padding: 10px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Water background */
    body {
        background-image: url("https://images.unsplash.com/photo-1586339949916-3e9457bef6d5?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }

    /* Glass container */
    .stApp {
        background: rgba(255, 255, 255, 0.75);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
        max-width: 1200px;
        margin: 2rem auto;
    }

    /* Typography */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        color: #1a2b4c;
    }

    h1, h2, h3 {
        color: #2c5da2;
        font-weight: 700;
    }

    .big-font {
        font-size: 28px !important;
        font-weight: 700;
        color: #114488;
        text-align: center;
    }

    .small-note {
        font-size: 13px;
        color: #444;
        text-align: center;
        margin-bottom: 10px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: #5fa3ec;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #347dd6;
    }

    /* Button styling */
    button[kind="primary"] {
        background-color: #4A90E2 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }

    button[kind="primary"]:hover {
        background-color: #357ABD !important;
    }

    /* Input boxes */
    .stSelectbox, .stSlider, .stRadio {
        background-color: rgba(255,255,255,0.9) !important;
        border-radius: 10px;
        padding: 5px;
    }

    /* DataFrame section tweaks */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 10px;
    }
    .produced-box {
        background: rgba(74, 144, 226, 0.1);
        padding: 10px;
        border-radius: 10px;
        margin-top: 10px;
        animation: fadeIn 2s ease-in-out;
        box-shadow: 0 0 5px rgba(74, 144, 226, 0.5);
    }

    .produced-box:hover {
        background: rgba(74, 144, 226, 0.15);
        transition: 0.3s;
        box-shadow: 0 0 10px rgba(74, 144, 226, 0.5);
    }

    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
# --- Custom Produced By section ---
st.markdown("""
    <div style="margin-top: 30px; padding: 10px; border-top: 1px solid #ccc;">
        <h4 style="color: #4A90E2;">👨‍💻 Produced By</h4>
        <div class="produced-box">
            <p><strong>Padasala Krish Veljibhai</strong></p>
            <p>Computer Engineering, Sem 7</p>
            <p style="font-size: 12px;">Shree Swami Atmanand Saraswati Institute of Technology</p>
            <p style="font-size: 12px;">Enrollment No: <b>220760107085</b></p>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("🔧 Settings")
    selected_country = st.selectbox("🌍 Choose a Country", countries)
    year_range = st.slider("📅 Prediction Year Range", 2024, 2040, (2025, 2035), step=1)
    compare_country = st.selectbox("🔍 Compare With", ["None"] + [c for c in countries if c != selected_country])

# --- Prediction Logic ---
future_years = np.arange(year_range[0], year_range[1] + 1, 5).reshape(-1, 1)
filtered = data[data['Country'] == selected_country]
X = filtered['Year'].values.reshape(-1, 1)
y = filtered['Water_Access_Percentage'].values
model = LinearRegression().fit(X, y)
predictions = model.predict(future_years)
r2 = model.score(X, y)

# --- Display Predictions ---
st.subheader(f"📊 Predictions for {selected_country}")
st.markdown("<div class='highlight'>", unsafe_allow_html=True)
for year, pred in zip(future_years.flatten(), predictions):
    st.markdown(f"- **{year}** → `{min(pred, 100):.2f}%`")
st.markdown("</div>", unsafe_allow_html=True)

# --- Reaching 100% ---
try:
    year_100 = int((100 - model.intercept_) / model.coef_[0])
    if year_100 <= 2040:
        st.success(f"🚀 {selected_country} may reach 100% access by **{year_100}**")
    else:
        st.warning("⚠️ 100% access may not be achieved by 2040.")
except:
    st.warning("⚠️ Unable to calculate 100% access year.")

# --- Growth and Confidence ---
col1, col2 = st.columns(2)
with col1:
    st.metric("📉 Model Confidence (R²)", f"{r2:.2f}")
with col2:
    if len(y) >= 2:
        growth = ((y[-1] - y[-2]) / y[-2]) * 100
        st.metric("📈 Recent Growth", f"{growth:.2f}%")

# --- Trend Graph ---
st.subheader("📈 Water Access Trend")
fig, ax = plt.subplots()
ax.scatter(X, y, color='skyblue', label='Actual Data')
ax.plot(future_years, predictions, color='limegreen', marker='o', label='Prediction')
ax.set_xlabel("Year")
ax.set_ylabel("Water Access (%)")
ax.set_title(f"{selected_country} Trend")
ax.set_ylim(0, 105)
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- Suggestions ---
with st.expander("💡 AI Suggestions to Improve Water Access"):
    latest_access = y[-1]
    if latest_access < 70:
        st.warning("🔸 Major efforts needed!")
        st.markdown("""
        - Launch rural water supply programs  
        - Promote rainwater harvesting  
        - Partner with NGOs for education  
        - Invest in waste treatment  
        """)
    elif 70 <= latest_access < 90:
        st.info("🔹 Moderate progress — needs scaling:")
        st.markdown("""
        - Expand piped systems  
        - Upgrade treatment plants  
        - Monitor water quality in urban areas  
        """)
    else:
        st.success("✅ Strong progress — maintain excellence:")
        st.markdown("""
        - Use AI-powered sensors  
        - Run awareness campaigns  
        - Maintain digital water monitoring  
        """)

# --- Country Comparison ---
if compare_country != "None":
    st.subheader(f"📊 {selected_country} vs {compare_country}")
    second = data[data['Country'] == compare_country]
    X2 = second['Year'].values.reshape(-1, 1)
    y2 = second['Water_Access_Percentage'].values
    model2 = LinearRegression().fit(X2, y2)
    preds2 = model2.predict(future_years)

    fig2, ax2 = plt.subplots()
    ax2.scatter(X, y, color='blue', label=f'{selected_country} Actual')
    ax2.plot(future_years, predictions, color='green', label=f'{selected_country} Prediction')
    ax2.scatter(X2, y2, color='red', label=f'{compare_country} Actual')
    ax2.plot(future_years, preds2, color='orange', label=f'{compare_country} Prediction')
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Water Access (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("Country Comparison")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

# --- Historical Table ---
with st.expander("📜 Historical Water Access Data"):
    st.dataframe(filtered[['Year', 'Water_Access_Percentage']].reset_index(drop=True))

# --- Top Countries by Access ---
with st.expander("🌐 Top 10 Countries by Access (2022)"):
    latest = data[data['Year'] == 2022].sort_values(by="Water_Access_Percentage", ascending=False)
    st.dataframe(latest[['Country', 'Water_Access_Percentage']].head(10))

# --- Download Prediction Data ---
download_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "Predicted Access (%)": [min(p, 100) for p in predictions]
})
csv = download_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download CSV", data=csv, file_name=f"{selected_country}_predictions.csv", mime="text/csv")

st.subheader("📣 Feedback")
st.markdown("""
    <div style="background-color: rgba(0, 0, 0, 0.4); padding: 15px; border-radius: 10px; color: black;">
""", unsafe_allow_html=True)

#feedback section
feedback = st.radio("How helpful was this prediction?", ["👍", "👌", "👎"])

if feedback:
    st.success("✅ Thank you for your feedback!")

st.markdown("</div>", unsafe_allow_html=True)

