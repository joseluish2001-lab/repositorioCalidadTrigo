import streamlit as st
import pandas as pd
import joblib

# Load the preprocessing objects and the model
try:
    one_hot_encoder = joblib.load('one_hot_encoder.pkl')
    standar_scaler = joblib.load('standar_scaler.pkl')
    Random_Forest_Regressor = joblib.load('Random_Forest_Regressor.pkl')
except FileNotFoundError:
    st.error("Error al cargar los archivos del modelo. Asegúrese de que 'one_hot_encoder.pkl', 'standar_scaler.pkl' y 'Random_Forest_Regressor.pkl' estén en el mismo directorio que el script de la aplicación.")
    st.stop()

st.title("Predicción de Fuerza de Trigo (Fuerza_W)")
st.write("Ingrese los detalles del grano para predecir la Fuerza_W.")

# Create input fields for the features
pais_origen = st.selectbox("País de Origen", ['USA', 'Argentina', 'Canada'])
region_productora = st.selectbox("Región Productora", ['Kansas', 'Buenos Aires', 'Saskatchewan', 'Manitoba', 'Cordoba', 'North Dakota'])

radiacion_mj_m2 = st.number_input("Radiación (MJ/m2)", min_value=0.0, value=18.0)
cenizas_pct = st.number_input("Cenizas (%)", min_value=0.0, value=0.5)
proteina_pct = st.number_input("Proteína (%)", min_value=0.0, value=13.0)
gluten_pct = st.number_input("Gluten (%)", min_value=0.0, value=28.0)
estabilidad_min = st.number_input("Estabilidad (min)", min_value=0.0, value=300.0)
p_l = st.number_input("P/L", min_value=0.0, value=0.8)


# Create a button to make predictions
if st.button("Predecir"):
    # Create a DataFrame from the input data
    input_data = pd.DataFrame([[
        pais_origen, region_productora, 
        radiacion_mj_m2, cenizas_pct, proteina_pct, gluten_pct, estabilidad_min, p_l
    ]],
    columns=[
        'Pais_Origen', 'Region_Productora', 
        'Radiacion_MJ_m2', 'Cenizas_%', 'Proteina_%', 'Gluten_%', 'Estabilidad_min', 'P_L'
    ])

    # Preprocess the input data

    # Apply one-hot encoder to categorical columns
    categorical_cols = ['Pais_Origen', 'Region_Productora']
    input_categorical = input_data[categorical_cols]
    input_encoded = one_hot_encoder.transform(input_categorical)

    # Get feature names from the one-hot encoder
    encoded_feature_names = list(one_hot_encoder.get_feature_names_out(categorical_cols))
    # Create a DataFrame from encoded features with names
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_feature_names, index=input_data.index)

    # Select numerical columns
    numerical_cols = ['Radiacion_MJ_m2', 'Cenizas_%', 'Proteina_%', 'Gluten_%', 'Estabilidad_min', 'P_L']
    input_numerical = input_data[numerical_cols]

    # Concatenate the encoded categorical features with the numerical features
    # Order: encoded first, then numerical
    input_processed = pd.concat([input_encoded_df, input_numerical], axis=1)

    # Define the expected feature order from the training data
    expected_feature_order = [
        'Pais_Origen_Argentina', 'Pais_Origen_Canada', 'Pais_Origen_USA',
        'Region_Productora_Buenos Aires', 'Region_Productora_Cordoba', 'Region_Productora_Kansas',
        'Region_Productora_Manitoba', 'Region_Productora_North Dakota', 'Region_Productora_Saskatchewan',
        'Radiacion_MJ_m2', 'Cenizas_%', 'Proteina_%', 'Gluten_%', 'Estabilidad_min', 'P_L'
    ]

    # Reindex to ensure the correct order and handle potentially missing categories by filling with 0
    input_processed = input_processed.reindex(columns=expected_feature_order, fill_value=0)

    # Apply standar scaler to the processed data
    input_scaled = standar_scaler.transform(input_processed)

    # Convert scaled numpy array back to DataFrame for consistency with column names
    input_scaled_df = pd.DataFrame(input_scaled, columns=expected_feature_order)

    # Make prediction
    prediction = Random_Forest_Regressor.predict(input_scaled_df)

    # Display the prediction
    st.success(f"Predicción de Fuerza_W: {prediction[0]:.2f}")
