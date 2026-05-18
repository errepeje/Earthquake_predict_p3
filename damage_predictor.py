from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Cargar datasets
X_train = pd.read_csv("train_values.csv", index_col="building_id")
y_train = pd.read_csv("train_labels.csv", index_col="building_id")

X_test = pd.read_csv("test_values.csv", index_col="building_id")

# Combinar train + labels
df = X_train.copy()
df["damage_grade"] = y_train["damage_grade"]

# Mostrar dimensiones
print("Train:", X_train.shape)
print("Labels:", y_train.shape)
print("Test:", X_test.shape)

df.head()
df.info()

# Verificar valores nulos
print("Valores nulos en train:")
print(X_train.isnull().sum())
print("\nValores nulos en test:")
print(X_test.isnull().sum())

# Verificar tipos de datos
print("Tipos de datos en train:")
print(X_train.dtypes)
print("\nTipos de datos en test:")
print(X_test.dtypes)

# Estadísticas descriptivas
print("Estadísticas descriptivas de train:")
print(X_train.describe())
print("\nEstadísticas descriptivas de test:")
print(X_test.describe())

# Visualizar la distribución de la variable objetivo
plt.figure(figsize=(10, 6))
plt.bar(y_train["damage_grade"].value_counts().sort_index().index, y_train["damage_grade"].value_counts().sort_index().values)
plt.title("Distribución de damage_grade")
plt.xlabel("damage_grade")
plt.ylabel("Frecuencia")
plt.show()


# preselección de características basada en el análisis exploratorio
selected_features = [
    "foundation_type",
    "area_percentage",
    "height_percentage",
    "count_floors_pre_eq",
    "land_surface_condition",
    "has_superstructure_cement_mortar_stone",
    "has_superstructure_mud_mortar_stone",
    "geo_level_1_id",
    "geo_level_2_id",
    "geo_level_3_id"
]



train_values_subset = X_train[selected_features]

# Visualizar relaciones entre características seleccionadas y la variable objetivo
#sns.pairplot(train_values_subset.join(y_train["damage_grade"]), hue="damage_grade")
#plt.show()

# preprocesamiento de datos: codificación de variables categóricas
train_values_subset = pd.get_dummies(train_values_subset)



# Random Forest subset
X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(train_values_subset, y_train["damage_grade"], test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_rf, y_train_rf)
y_pred_rf = rf_model.predict(X_val_rf)
f1_rf = f1_score(y_val_rf, y_pred_rf, average="micro")
print(f"Random Forest Micro F1-score: {f1_rf:.4f}")

# Random Forest todas las características

train_values_encoded = pd.get_dummies(X_train)
test_values_encoded = pd.get_dummies(X_test)

X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(
    train_values_encoded,
    y_train["damage_grade"],
    test_size=0.2,
    stratify=y_train["damage_grade"],
    random_state=42
)
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_rf, y_train_rf)
y_pred_rf = rf_model.predict(X_val_rf)
f1_rf = f1_score(y_val_rf, y_pred_rf, average="micro")
print(f"Random Forest Micro F1-score: {f1_rf:.4f}")

# XGBoost 

y_train_aux = y_train.copy()
# XGBoost requiere clases 0,1,2 en lugar de 1,2,3 asi que dependemos de una transformación simple restando 1 a las etiquetas y guardandolo como auxiliar
y_train_aux["damage_grade"] = y_train_aux["damage_grade"] - 1
X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split( train_values_subset, y_train_aux["damage_grade"], test_size=0.2, random_state=42 )

xgb_model = XGBClassifier(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)
xgb_model.fit(X_train_xgb, y_train_xgb)
y_pred_xgb = xgb_model.predict(X_val_xgb)
f1_xgb = f1_score(y_val_xgb, y_pred_xgb, average="micro")
print(f"XGBoost Micro F1-score: {f1_xgb:.4f}")



# Preparar test
test_values_subset = X_test[selected_features]
# preprocesamiento de datos: codificación de variables categóricas
test_values_subset = pd.get_dummies(test_values_subset)
# Entrenar modelo final con todos los datos
xgb_model.fit(train_values_subset, y_train_aux["damage_grade"])

# Predecir test
test_preds = xgb_model.predict(test_values_subset)
test_preds = test_preds + 1  # volver a 1,2,3

# Crear submission
submission = pd.DataFrame({
    "building_id": X_test.index,
    "damage_grade": test_preds
})

submission.to_csv("submission.csv", index=False)

