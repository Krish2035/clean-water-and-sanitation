import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# ---------- Custom CSS -------------
st.markdown("""
    <style>
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f0f9ff;
            border-right: 2px solid #cce7ff;
        }

        /* App background */
        .main {
            background: linear-gradient(to right, #ffffff, #f2f9ff);
            padding: 15px;
            font-family: 'Segoe UI', sans-serif;
            color: #222 !important;  /* universal text color */
        }

        /* Universal text color */
        h1, h2, h3, h4, h5, h6, p, span, div, label {
            color: #222 !important;
        }

        /* Section boxes */
        .stSubheader, .stMarkdown, .stDataFrame, .stDownloadButton {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 15px;
            color: #222 !important;
        }

        /* Buttons and widgets */
        .stButton>button, .stDownloadButton>button {
            background-color: #007acc;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
        }

        .stButton>button:hover, .stDownloadButton>button:hover {
            background-color: #005fa3;
        }

        /* Input widget labels */
        .css-1aumxhk, .css-1n76uvr {
            color: #222 !important;
        }
    </style>
""", unsafe_allow_html=True)


# ---------- Data and Sidebar ----------
data = pd.read_csv("expanded_country_water_access_data.csv")
countries = sorted(data['Country'].unique())

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/UN-Water_logo.svg/2560px-UN-Water_logo.svg.png",
    use_container_width=True
)
st.sidebar.title("🌊 Water Access AI")
menu = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📈 Predictions", "💡 Suggestions", "🆚 Compare", "🌐 Top 10", "📥 Download"]
)

selected_country = st.sidebar.selectbox("Select Country", countries)
year_range = st.sidebar.slider("Year Range", 2024, 2040, (2025, 2035), step=1)
future_years = np.arange(year_range[0], year_range[1]+1, 5).reshape(-1, 1)

# Prepare prediction
filtered = data[data['Country'] == selected_country]
X = filtered['Year'].values.reshape(-1, 1)
y = filtered['Water_Access_Percentage'].values
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(future_years)

# ---------- Pages ----------
if menu == "🏠 Home":
    st.markdown("""
        <div style='text-align: center; padding: 10px; border-bottom: 2px solid #ccc;'>
            <h2>🔮 Clean Water Access Prediction using AI</h2>
            <h4>Padasala Krish Veljibhai • Computer Engineering Sem 7</h4>
            <p>Shree Swami Atmanand Saraswati Institute of Technology | Institute Code: 076 | Branch Code: 07</p>
            <p>Enrollment Number: <strong>220760107085</strong></p>
        </div>
    """, unsafe_allow_html=True)
    st.write("🌍 Select a country from the sidebar to predict clean water access for custom future years.")

elif menu == "📈 Predictions":
    st.subheader(f"📊 Predictions for {selected_country}")
    for year, pred in zip(future_years.flatten(), predictions):
        st.markdown(f"- *{year}* → {min(pred, 100):.2f}%")

    fig, ax = plt.subplots()
    ax.scatter(X, y, color='skyblue', label='Actual Data')
    ax.plot(future_years, predictions, color='limegreen', marker='o', label='Prediction')
    ax.set_xlabel("Year")
    ax.set_ylabel("Water Access (%)")
    ax.set_title(f"Water Access Trend – {selected_country}")
    ax.set_ylim(0, 105)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

elif menu == "💡 Suggestions":
    st.subheader("💡 Suggestions to Improve Clean Water Access")
    latest_access = y[-1]

    if latest_access < 70:
        st.warning("🔸 Major efforts needed!")
        st.markdown("""
        - Launch rural water supply programs  
        - Promote rainwater harvesting  
        - NGO & govt collaboration for awareness  
        - Improve sewage & waste infrastructure
        """)
    elif 70 <= latest_access < 90:
        st.info("🔹 Moderate progress — needs reinforcement:")
        st.markdown("""
        - Expand piped water systems  
        - Upgrade filtration plants  
        - Increase quality monitoring
        """)
    else:
        st.success("✅ High access — focus on smart solutions:")
        st.markdown("""
        - AI-based water sensors  
        - Real-time contamination alerts  
        - Public education campaigns  
        """)

elif menu == "🆚 Compare":
    st.subheader("🔍 Compare with Another Country")
    compare_country = st.selectbox("Compare With", ["None"] + [c for c in countries if c != selected_country])

    if compare_country != "None":
        second_data = data[data['Country'] == compare_country]
        X2 = second_data['Year'].values.reshape(-1, 1)
        y2 = second_data['Water_Access_Percentage'].values
        model2 = LinearRegression()
        model2.fit(X2, y2)
        predictions2 = model2.predict(future_years)

        fig2, ax2 = plt.subplots()
        ax2.scatter(X, y, color='skyblue', label=f'{selected_country} Actual')
        ax2.plot(future_years, predictions, color='green', label=f'{selected_country} Prediction')
        ax2.scatter(X2, y2, color='salmon', label=f'{compare_country} Actual')
        ax2.plot(future_years, predictions2, color='red', label=f'{compare_country} Prediction')
        ax2.set_title(f"{selected_country} vs {compare_country}")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Water Access (%)")
        ax2.set_ylim(0, 105)
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

elif menu == "🌐 Top 10":
    st.subheader("🌐 Top 10 Countries by Access (2022)")
    latest = data[data['Year'] == 2022].sort_values(by="Water_Access_Percentage", ascending=False)
    st.dataframe(latest[['Country', 'Water_Access_Percentage']].head(10))

elif menu == "📥 Download":
    st.subheader("📥 Download Prediction Data")
    download_df = pd.DataFrame({
        "Year": future_years.flatten(),
        "Predicted Access (%)": [min(p, 100) for p in predictions]
    })
    csv = download_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name=f"{selected_country}_water_predictions.csv", mime="text/csv")
