import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, r2_score

# 1. ЗАГРУЗКА
df = pd.read_csv('./dataset/60k.csv')

print(f"Всего строк до чистки: {len(df)}")

df = df.drop_duplicates(subset=['id'])
df = df[(df['price'] > 6_000_000) & (df['price'] < 100_000_000)]
df = df[(df['area_total'] > 15) & (df['area_total'] < 300)]
print(f"Строк после чистки: {len(df)}")


# Флаг: первый или последний этаж (это сильно влияет на цену)
df['is_first_floor'] = (df['floor'] == 1).astype(int)
df['is_last_floor'] = (df['floor'] == df['total_floors']).astype(int)

df['area_living'] = df['area_living'].fillna(df['area_total'] * 0.6) 
df['area_kitchen'] = df['area_kitchen'].fillna(df['area_total'] * 0.2)
df['living_share'] = df['area_living'] / df['area_total']       # Какую часть занимает жил площадь
df['kitchen_share'] = df['area_kitchen'] / df['area_total']     # Какую часть занимает кухня
df['floor_ratio'] = df['floor'] / df['total_floors']            # Высоко ли этаж относительно дома

cat_features = ['material', 'metro', 'address'] 
text_features = ['address'] 

for col in cat_features:
    df[col] = df[col].fillna('unknown').astype(str)

num_features = ['rooms', 'area_total', 'area_living', 'area_kitchen', 
                'floor', 'total_floors', 'year_built', 'time_to_metro', 
                'is_first_floor', 'is_last_floor', 
                'living_share', 'kitchen_share', 'floor_ratio']

# 4. ЛОГАРИФМИРОВАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
X = df[cat_features + num_features]
y = np.log1p(df['price']) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. ОБУЧЕНИЕ
# Используем text_features для адреса
model = CatBoostRegressor(
    iterations=3500, 
    learning_rate=0.05,
    depth=8,
    loss_function='MAE',
    eval_metric='MAE',
    l2_leaf_reg=5,       # Регуляризация переобучения
    verbose=200,
    random_seed=42
)

print("Начинаем обучение...")
train_pool = Pool(X_train, y_train, cat_features=cat_features, text_features=text_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features, text_features=text_features)

model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=200)

# 6. ОЦЕНКА (Возвращаем цены из логарифма обратно в рубли)
preds_log = model.predict(test_pool)
preds_real = np.expm1(preds_log) 
y_test_real = np.expm1(y_test)

mae = mean_absolute_error(y_test_real, preds_real)
r2 = r2_score(y_test_real, preds_real)

print("\n" + "="*40)
print(f"REAL MAE (Ошибка в рублях): {mae:,.0f} руб.")
print(f"R2 Score (Точность): {r2:.4f}")
print("="*40)

# 7. ВАЖНОСТЬ ПРИЗНАКОВ
print("\nТоп факторов:")
print(model.get_feature_importance(prettified=True).head(7))
model.save_model("./model/price_model_v2.cbm")