import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

cbr=CatBoostRegressor()
model = cbr.load_model("./model/price_model_v2.cbm")
print(model.get_feature_importance(prettified=True))
new_flat_data = {
    'material': 'panel',
    'metro': 'Новокосино',
    'address': 'Россия, Москва, Новокосинская улица, 15к3',
    'rooms': 3.0,
    'area_total': 85.0,
    'area_living': 76.0,
    'area_kitchen': 8.0,
    'floor': 14,
    'total_floors': 17,
    'year_built': 2005,
    'time_to_metro': 12.0,
    'is_first_floor': 0,
    'is_last_floor': 0,
    'living_share': 76.0,
    'kitchen_share': 8.0,
    'floor_ratio': (14//17)
}

new_df = pd.DataFrame([new_flat_data])
pred_log = model.predict(new_df)
price = np.expm1(pred_log)[0]
print(f"\nЦена тестовой квартиры: {price:,.0f} руб.")