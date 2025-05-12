
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
from colorama import Fore, Style, init as colorama_init
import shap
import matplotlib.pyplot as plt
import uuid
import os

colorama_init(autoreset=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("traffic-ai")

model = joblib.load("RandomForestTrafficModel.pkl")
encoder_dict = joblib.load("LabelEncoders.pkl")
trained_features = joblib.load("TrainedFeatures.pkl")

app = FastAPI(title="Sydney Traffic AI System")
app.mount("/static", StaticFiles(directory="static"), name="static")

class TrafficInput(BaseModel):
    year: int
    period: str
    classification_type: str
    station_id: str
    post_code: str
    state: str
    suburb_x: str
    road_name_x: str
    cardinal_direction_name: str

@app.get("/status")
def status():
    return {"message": "API is running!"}

@app.post("/predict")
def predict(data: TrafficInput):
    try:
        input_df = pd.DataFrame([data.dict()])

        logger.info(Fore.CYAN + "Encoding categorical columns:")
        categorical_cols = [
            'period', 'classification_type', 'cardinal_direction_name',
            'suburb_x', 'road_name_x'
        ]

        for col in categorical_cols:
            le = encoder_dict[col]
            val = input_df[col].iloc[0]
            if val in le.classes_:
                encoded_val = le.transform([val])[0]
                input_df[col] = encoded_val
                logger.info(Fore.GREEN + f"✅ Encoded '{val}' to {encoded_val} for '{col}'")
            else:
                logger.warning(Fore.YELLOW + f"⚠️ '{val}' not seen before in column '{col}', set to -1")
                input_df[col] = -1

        for col in trained_features:
            if col not in input_df.columns:
                input_df[col] = -1

        input_df = input_df[trained_features]
        prediction = model.predict(input_df)[0]

        if prediction < 3000:
            traffic_level = "Low traffic"
        elif prediction < 7000:
            traffic_level = "Moderate traffic"
        else:
            traffic_level = "High traffic"


        df = pd.read_csv("cleaned_traffic_data.csv")
        suburbs = df["suburb"].dropna().unique()

        results = []
        for suburb in suburbs:
            payload = {
                "year": data.year,
                "period": "ALL DAYS",
                "classification_type": "ALL VEHICLES",
                "station_id": -1,
                "post_code": "0000",
                "state": "NSW",
                "suburb_x": suburb,
                "road_name_x": "Unknown",
                "cardinal_direction_name": "BOTH"
            }

            temp_df = pd.DataFrame([payload])
            for col in encoder_dict:
                le = encoder_dict[col]
                val = temp_df[col].iloc[0]
                temp_df[col] = le.transform([val])[0] if val in le.classes_ else -1

            for col in trained_features:
                if col not in temp_df.columns:
                    temp_df[col] = -1

            temp_df = temp_df[trained_features]
            pred = model.predict(temp_df)[0]

            results.append((suburb, pred))

        results.sort(key=lambda x: x[1], reverse=True)
        total = len(results)
        suburb_rank = next((i+1 for i, (suburb, val) in enumerate(results)
                            if suburb.strip().upper() == data.suburb_x.strip().upper()), -1)

        return {
            "predicted_traffic_count": int(prediction),
            "context": traffic_level,
            "rank": suburb_rank,
            "total_suburbs": total
        }

    except Exception as e:
        logger.error(Fore.RED + f"🔥 ERROR during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/explain")
def explain(data: TrafficInput):
    try:
        input_df = pd.DataFrame([data.dict()])
        categorical_cols = [
            'period', 'classification_type', 'cardinal_direction_name',
            'suburb_x', 'road_name_x'
        ]

        for col in categorical_cols:
            le = encoder_dict[col]
            val = input_df[col].iloc[0]
            if val in le.classes_:
                input_df[col] = le.transform([val])[0]
            else:
                input_df[col] = -1

        for col in trained_features:
            if col not in input_df.columns:
                input_df[col] = -1

        input_df = input_df[trained_features]

        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)

        top_indices = abs(shap_values.values[0]).argsort()[-5:][::-1]
        top_features = [
            {"feature": input_df.columns[i], "impact": round(shap_values.values[0][i], 2)}
            for i in top_indices
        ]

        os.makedirs("static", exist_ok=True)
        plot_id = str(uuid.uuid4())
        shap.plots.beeswarm(shap_values, show=False, color=plt.get_cmap("magma"))
        plt.gcf().set_size_inches(8, 5)
        plot_path = f"static/shap_plot_{plot_id}.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.clf()

        return {
            "top_features": top_features,
            "shap_plot_url": f"/static/shap_plot_{plot_id}.png"
        }

    except Exception as e:
        logger.error(Fore.RED + f"🔥 SHAP explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

