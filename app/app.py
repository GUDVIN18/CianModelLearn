import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from catboost import CatBoostRegressor
from shemas import  PriceResponse, ApartmentFeatures


app = FastAPI(title="Real Estate Price Predictor")
cbr=CatBoostRegressor()
model = cbr.load_model("../model/price_model_v2.cbm")

@app.post(
    "/home-price", 
    name="ИИ расчет стомости жилья в Москве",
    response_model=PriceResponse
)
def predict_price(apartment: ApartmentFeatures):
    try:
        new_df = pd.DataFrame([apartment.model_dump()])
        pred_log = model.predict(new_df)
        price = round(np.expm1(pred_log)[0])

        return PriceResponse(
            predicted_price=price,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)