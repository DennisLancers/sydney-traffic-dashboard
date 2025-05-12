import streamlit as st
import pandas as pd
import requests
import json
import io
import re

with open("suburb_station_postcode_mapping.json", "r") as f:
    mapping = json.load(f)

st.set_page_config(page_title="Sydney Traffic AI System", layout="centered")
st.sidebar.markdown("## ℹ️ About")
st.sidebar.info("""
Sydney Traffic AI Dashboard
Built for predecting traffic trends in Sydney suburbs.
© 2025 Steven.E.NASER
""")
st.title(" Sydney Traffic AI System")

suburb = st.selectbox("Select Suburb", sorted(mapping.keys()))

station_id_fix = mapping[suburb]["station_id"]
station_id = re.sub(r"[^\d]", "", station_id_fix) 
road_name_x = mapping[suburb]["road_name_x"]
post_code = mapping[suburb]["post_code"]


st.markdown("###  Prediction Inputs")

cl1, cl2 = st.columns(2)

with cl1:
    year = st.number_input(" Year", min_value=2020, max_value=2025, value=2025)
    period = st.selectbox(" Period", [
        'ALL DAYS', 'AM PEAK', 'OFF PEAK', 'PM PEAK', 'PUBLIC HOLIDAYS',
        'SCHOOL HOLIDAYS', 'WEEKDAYS', 'WEEKENDS'
    ])

with cl2:
    classification_type = st.selectbox(" Vehicle Type", [
        'ALL VEHICLES', 'HEAVY VEHICLES', 'LIGHT VEHICLES', 'UNCLASSIFIED', 'PASSENGER VEHICLES'
    ])
    cardinal_direction_name = st.selectbox(" Direction", [
        'BOTH', 'EAST', 'NORTH', 'SOUTH', 'WEST'
    ])

st.markdown("###  Prediction Output")



if st.button("Prediction"):
    payload = {
        "year": year,
        "period": period,
        "classification_type": classification_type,
        "station_id": station_id,
        "post_code": post_code,
        "state": "NSW",
        "suburb_x": suburb,
        "road_name_x": road_name_x,
        "cardinal_direction_name": cardinal_direction_name
    }

    response = response = requests.post("https://my-fastapi-backend.onrender.com/predict", json=payload)


    if response.status_code == 200:
        result = response.json()
        st.success(f" **Traffic Count:** {result['predicted_traffic_count']}")
        st.markdown(f"<span style='font-size: 20px; font-weight: bold;'> {result['context']}</span>", unsafe_allow_html=True)
        st.markdown(
            f" <b>{suburb.title()}</b> is the <b>{result['rank']}</b> out of <b>{result['total_suburbs']}</b> suburbs in <b>{year}</b>.",
            unsafe_allow_html=True
        )

        PayloadOutput = pd.DataFrame([payload])
        PayloadOutput["Predicted Traffic"] = result["predicted_traffic_count"]
        PayloadOutput["Traffic Context"] = result["context"]
        CsvOuter = io.StringIO()
        PayloadOutput.to_csv(CsvOuter, index=False)
        st.download_button(
            label="  Download Traffic in CSV form",
            data = CsvOuter.getvalue(),
            file_name="traffic_prediction.csv",
            mime="text/csv"
        )
    else:
        st.error("Prediction failed.")

st.markdown("###  SHAP Explanation")

if st.button(" Explain Predictions"):
    payload = {
        "year": year,
        "period": period,
        "classification_type": classification_type,
        "station_id": station_id,
        "post_code": post_code,
        "state": "NSW",
        "suburb_x": suburb,
        "road_name_x": road_name_x,
        "cardinal_direction_name": cardinal_direction_name
    }

    explain_response = response = requests.post("https://my-fastapi-backend.onrender.com/predict", json=payload)

    if explain_response.status_code == 200:
        result = explain_response.json()
        st.subheader("🔍 Top Feature Impacts")
        for feature in result["top_features"]:
            st.write(f"**{feature['feature']}**: {feature['impact']}")
        st.image("https://my-fastapi-backend.onrender.com" + result["shap_plot_url"])
    else:
        st.error("Explanation failed.")

st.markdown("### Suburb Location")

try:
    station_data = pd.read_csv("Traffic_Volume_Viewer_-_Data_for_All_Years (2).csv")
    station_data = station_data.drop_duplicates(subset=["station_id"])
    station_data = station_data.rename(columns={
        "wgs84_latitude": "lat",
        "wgs84_longitude": "lon"
    })
    station_data = station_data.dropna(subset=["lat", "lon"])
    filtered_df = station_data[station_data["suburb"].str.strip().str.upper() == suburb.strip().upper()]

    if not filtered_df.empty:
        st.map(filtered_df)
    else:
        st.warning("No data existed for the selected suburb.")
except Exception as e:
    st.warning(f"Unable to load map: {e}")
        
try:
    trend_df = pd.read_csv("cleaned_traffic_data.csv")
    trend_df = trend_df[trend_df["suburb"].str.strip().str.upper() == suburb.strip().upper()]
    yearly_trend = trend_df.groupby("year")["traffic_count"].mean().reset_index()
    st.markdown("### Traffic Plot Using Selected Suburb")
    st.line_chart(yearly_trend.set_index("year"))
except Exception as e:
    st.warning("Can't load data: " + str(e))
st.markdown("### Top 5 Congested Suburbs")

try:
    df = pd.read_csv("cleaned_traffic_data.csv")
    top5 = df[df['year'] == year].groupby("suburb")["traffic_count"].mean().reset_index()
    top5 = top5.sort_values(by="traffic_count", ascending=False).head(5)
    st.bar_chart(top5.set_index("suburb"))
except Exception as e:
    st.warning("Can't load data: " + str(e))
        
st.markdown("---")
st.caption("Sydney Traffic AI System © 2025 | Built by Steven.E.NASER")
