import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
players = pd.read_csv("C:\\Users\\gabol\\Desktop\\TFM_NBA\\DF-PLAYERS.csv", low_memory=False)

# Remove leading/trailing spaces from column names
players.columns = players.columns.str.strip()

# Convert the 'Date' column to datetime
players['Date'] = pd.to_datetime(players['Date'], errors='coerce')

# List of columns to process
columnas_procesar = ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', '+/-']

# Convert columns to numeric, errors='coerce' will replace non-numeric values with NaN
for col in columnas_procesar:
    players[col] = pd.to_numeric(players[col], errors='coerce')

# Calculate the mean for each player and fill NaN values with the mean
promedio_por_jugador = players.groupby('Player Name')[columnas_procesar].transform(lambda x: x.fillna(x.mean()))

# Create a copy of the DataFrame and fill NaN values
df_asis = players.copy()
df_asis[columnas_procesar] = df_asis[columnas_procesar].fillna(promedio_por_jugador)

# Function to calculate cumulative mean
def calcular_media_acumulativa(df):
    new_columns = []
    for col in columnas_procesar:
        new_col = col + '_mean'
        new_columns.append(df[col].expanding().mean().rename(new_col))
    new_df = pd.concat(new_columns, axis=1)
    return new_df

# Streamlit app
st.title("NBA Player Stats Explorer")
st.write("Esta aplicación predice las estadísticas de jugadores de la NBA en los playoffs utilizando diferentes modelos de regresión.")

# Sidebar input
st.sidebar.header("User Input Features")
player_selected = st.sidebar.selectbox("Jugador", options=players['Player Name'].unique())
column_selected = st.sidebar.selectbox("Columna a predecir", options=columnas_procesar)

# Filter data based on selection
filtered_data = players[players['Player Name'] == player_selected]

# Check if there's data for the selected player
if filtered_data.empty:
    st.write(f"No hay datos disponibles para {player_selected}.")
else:
    st.header(f"Datos Filtrados para {player_selected}")
    st.dataframe(filtered_data)

    # Calculate cumulative mean for the selected player
    new_df = calcular_media_acumulativa(filtered_data)

    # Select the desired columns
    columnas_disponibles = [col for col in columnas_procesar if col + '_mean' in new_df.columns]
    luka_work = new_df[[col + '_mean' for col in columnas_disponibles]]

    # Split into predictor and target variables
    X, y = luka_work.drop(columns=[column_selected + '_mean']).values, luka_work[column_selected + '_mean'].values
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.3)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train2 = imputer.fit_transform(X_train2)
    X_test2 = imputer.transform(X_test2)
    y_train2 = imputer.fit_transform(y_train2.reshape(-1, 1)).ravel()
    y_test2 = imputer.transform(y_test2.reshape(-1, 1)).ravel()

    # Train and evaluate DecisionTreeRegressor
    best_mae = float('inf')
    best_depth = None
    for depth in range(1, 250):
        clf = DecisionTreeRegressor(max_depth=depth)
        clf.fit(X_train2, y_train2)
        tpredictions = clf.predict(X_test2)
        t_error = np.abs(tpredictions - y_test2)
        mae = t_error.mean()
        if mae < best_mae:
            best_mae = mae
            best_depth = depth

    clf = DecisionTreeRegressor(max_depth=best_depth)
    clf.fit(X_train2, y_train2)
    tree_predictions = clf.predict(X_test2)

    # Train and evaluate LinearRegression
    reg = LinearRegression()
    reg.fit(X_train2, y_train2)
    lr_predictions = reg.predict(X_test2)
    lr_mse = mean_squared_error(y_test2, lr_predictions)
    lr_r2 = r2_score(y_test2, lr_predictions)

    # Train and evaluate RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train2, y_train2)
    rf_predictions = rf.predict(X_test2)
    rf_mse = mean_squared_error(y_test2, rf_predictions)
    rf_r2 = r2_score(y_test2, rf_predictions)

    st.header("Evolución del Jugador a lo largo del tiempo")
    fig, ax = plt.subplots(figsize=(12, 8))
    for col in columnas_disponibles:
        ax.plot(new_df.index, new_df[col + '_mean'], label=col + '_mean')
    ax.set_xlabel("Partidos")
    ax.set_ylabel("Valores")
    ax.set_title(f"Evolución de las Estadísticas de {player_selected}")
    ax.legend()
    st.pyplot(fig)

    st.header("Resultados del Modelo")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Mejor profundidad (Decision Tree)", value=best_depth)
        st.metric(label="MAE (Decision Tree)", value=f"{best_mae:.2f}")
    with col2:
        st.metric(label="MSE (Linear Regression)", value=f"{lr_mse:.2f}")
        st.metric(label="R2 (Linear Regression)", value=f"{lr_r2:.2f}")
    with col3:
        st.metric(label="MSE (Random Forest)", value=f"{rf_mse:.2f}")
        st.metric(label="R2 (Random Forest)", value=f"{rf_r2:.2f}")

    st.header("Comparación de Predicciones")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(y_test2, label="Valores Reales", linewidth=2)
    ax.plot(tree_predictions, label="Predicciones Decision Tree", linestyle='--', linewidth=2)
    ax.plot(lr_predictions, label="Predicciones Linear Regression", linestyle=':', linewidth=2)
    ax.plot(rf_predictions, label="Predicciones Random Forest", linestyle='-.', linewidth=2)
    ax.legend()
    st.pyplot(fig)

    st.header("Distribución de Errores")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    tree_error_distribution = np.abs(tree_predictions - y_test2)
    sns.histplot(tree_error_distribution, kde=True, ax=ax1, color='blue')
    ax1.set_title("Decision Tree")

    lr_error_distribution = np.abs(lr_predictions - y_test2)
    sns.histplot(lr_error_distribution, kde=True, ax=ax2, color='green')
    ax2.set_title("Linear Regression")

    rf_error_distribution = np.abs(rf_predictions - y_test2)
    sns.histplot(rf_error_distribution, kde=True, ax=ax3, color='red')
    ax3.set_title("Random Forest")

    st.pyplot(fig)

    st.header("Predicciones de los Modelos")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric(label="Última Predicción (Decision Tree)", value=f"{tree_predictions[-1]:.2f}")
    with col5:
        st.metric(label="Última Predicción (Linear Regression)", value=f"{lr_predictions[-1]:.2f}")
    with col6:
        st.metric(label="Última Predicción (Random Forest)", value=f"{rf_predictions[-1]:.2f}")
