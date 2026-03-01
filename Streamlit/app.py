import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Funciones as f

# 1. Configuración y Carga de Datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv('ObesityDataSet_clean.csv')
    return df

# --- NAVEGACIÓN LATERAL ---
st.logo("Streamlit/imagen.png")
st.sidebar.title("Navegación")
seccion = st.sidebar.radio("Ir a:", ["Introducción",
                                    "Presentación de Datos",
                                    "Elección de las carácteristicas y técnicas",
                                    "Modelo de elección",
                                    "Ajuste de Parámetros",
                                    "Cuestionario de Salud",
                                    "Aplicación al mundo real",
                                    "Desafios y aprendizajes",
                                    "Motivación"])

# --- CONFIGURACIÓN DE ESTILO ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #E3F2FD;
    }
    </style>
    """,
    unsafe_allow_html=True)

# --- SECCIÓN 1: INTRODUCCIÓN ---
if seccion == "Introducción":
    st.title("Estimación de los niveles de obesidad")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("¿Qué es la obesidad?")
        st.write("La obesidad es un problema de salud que se caracteriza por un exceso de grasa corporal que puede afectar negativamente la salud.")
    with col2:
        st.subheader("¿Por qué es importante prevenirla?")
        st.write("La obesidad está asociada con un mayor riesgo de enfermedades crónicas como diabetes, hipertensión y enfermedades cardíacas.")
    st.image("Streamlit/imagen2.png")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Objetivo del Proyecto")
        st.write("El objetivo de este proyecto es desarrollar un modelo de machine learning que pueda predecir los niveles de obesidad en individuos basándose en su condición física y hábitos de vida.")
    with col2:
        st.subheader("¿Por qué es importante?")
        st.write("Porque permite la prevención y el tratamiento temprano de la obesidad, lo que puede mejorar la calidad de vida.")
    
# --- SECCIÓN 2: PRESENTACIÓN ---
if seccion == "Presentación de Datos":
    st.title("Análisis de la Base de Datos de Obesidad")
    df = cargar_datos()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Muestra de Datos")
        st.dataframe(df.head())
        
    with col2:
        st.subheader("Distribución de Niveles de Obesidad")
        orden_categorias = [
        "Peso insuficiente", 
        "Peso normal", 
        "Sobrepeso nivel I", 
        "Sobrepeso nivel II", 
        "Obesidad tipo I", 
        "Obesidad tipo II", 
        "Obesidad tipo III"
        ]
        fig = px.histogram(df, x='nivel_obesidad', color='nivel_obesidad',
        category_orders={'nivel_obesidad': orden_categorias},
        color_discrete_sequence=px.colors.sequential.Viridis)
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución por Género")
        fig = px.pie(df, names='género', hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Relación Altura vs Peso")
        fig2 = px.scatter(df, x="altura", y="peso", color="nivel_obesidad", size="altura")
        st.plotly_chart(fig2, use_container_width=True)
    st.divider() 
    
    st.subheader("Relación Edad vs Peso")
    fig2 = px.scatter(df, x="edad", y="peso", color="nivel_obesidad", size="altura")
    st.plotly_chart(fig2, use_container_width=True)
    st.write("Se puede observar en este estudio que las personas que se encuentran en el nivel de Obesidad tipo III, tienen entre 18 y 26 años.")
    st.divider()
    

# --- SECCIÓN 3: ELECCIÓN DE CARACTERÍSTICAS Y TÉCNICAS ---
if seccion == "Elección de las carácteristicas y técnicas":
    st.title("Elección de las carácteristicas y técnicas")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlación entre variables")
        df = pd.read_csv('ObesityDataSet_clean.csv')
        df_nuevo=f.traspaso_estudio(df)
        corr=np.abs(df_nuevo.corr())
        # Set up mask for triangle representation
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(10, 10))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask,  vmax=1,square=True, linewidths=.5, cbar_kws={"shrink": .5},annot = corr)
        st.pyplot(f)
    with col2:
        st.write("Se puede observar que las variables que tienen una correlación más débil con el nivel de obesidad son:")
        st.write("- género")
        st.write("- NCP (Número de comidas principales por día)")
        st.write("- fuma (¿Fuma o no fuma?)")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Se decidió usar las técnicas de:")
        st.write("- Label encoding para transformar las variables categóricas en numéricas.")
        st.write("- Feature Scaling para ajustar el rango de diferentes variables para que estén en una escala similar.")
    
    with col2:
        st.write("Beneficios:")
        st.write("- Puede ayudar a mejorar el rendimiento y la velocidad de convergencia de varios algoritmos de aprendizaje automático.")
        st.write("- Evita que las variables con valores más grandes dominen las variables con valores más pequeños.")

# --- SECCIÓN 4: MODELO DE ELECCIÓN ---
if seccion == "Modelo de elección":
    st.title("Modelo de elección")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Se decidió usar el modelo de gradient boosting Classifier")
        st.image("Streamlit/gradient.png")
        
    with col2:
        st.write("¿En que consiste?")
        st.write("Los errores desempeñan un papel fundamental en cualquier algoritmo de aprendizaje automático. "
            "Existen dos tipos principales de errores: error de sesgo y error de varianza. "
            "Este modelo nos ayuda a minimizar el error de sesgo. "
            "La idea principal es construir modelos secuencialmente, y estos modelos subsiguientes "
            "intentan reducir los errores del modelo anterior. "
            "Pero ¿cómo lo hacemos? ¿Cómo reducimos el error? "
            "Construyendo un nuevo modelo a partir de los errores o residuos del modelo anterior.")


# --- SECCIÓN 5: AJUSTE DE PARÁMETROS ---
if seccion == "Ajuste de Parámetros":
    st.title("Ajuste de Parámetros")
    
    st.subheader("Configuración del Modelo")
    
    modelos = {
        "Modelos": [
        "Random Forest", "Gradient Boosting", 
        "Decision Tree"
    ],
        "Accuracy (sin ajustar parámetros)": [0.94, 0.91, 0.87],
        "Accuracy (con ajustar parámetros)": [0.96, 0.97, 0.94],
        "F1-Score (sin ajustar parámetros)": [0.94, 0.92, 0.82],
        "F1-Score (con ajustar parámetros)": [0.94, 0.96, 0.94]
    }
    df_modelos = pd.DataFrame(modelos)
    st.dataframe(df_modelos, hide_index=True)
    st.divider()
    
    st.image("Streamlit/matriz.png")
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        st.subheader("Configuración del Modelo")
        # Creamos un diccionario con los parámetros
        params = {
        "Parámetro": [
        "n_estimators", "max_depth", "learning_rate", 
        "cv"
    ],
        "Valor": [300, 5, 0.2, 5]
        }
        df_params = pd.DataFrame(params)
        st.dataframe(df_params, hide_index=True)

    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("Se ajustaron los parámetros para optimizar el rendimiento del modelo.")
        st.write("Se usó la técnica de grid search para encontrar los mejores parámetros.") 

# --- SECCIÓN 6: CUESTIONARIO---
elif seccion == "Cuestionario de Salud":
    st.title("Predicción de Nivel de Obesidad")
    st.write("Introduce tus datos para comparar con nuestra base de datos:")
    
    with st.form("user_data"):
        # Ejemplo de campos (ajusta a tus columnas reales)
        edad = st.number_input("Edad", min_value=1, max_value=100, value=25)
        estatura = st.number_input("Estatura (m)", min_value=0.5, value=1.70)
        peso = st.number_input("Peso (kg)", min_value=10.0, value=70.0)
        f_con_sobrepeso = st.selectbox("¿Antecedentes familiares con sobrepeso?", ["si", "no"])
        FAVC = st.selectbox("¿Comes alimentos altos en calorías frecuentemente?", ["si", "no"])
        FCVC = st.slider("¿Nivel de verduras en tus comidas?", 0.0, 3.0, 1.0)
        CAEC = st.selectbox("¿Comes entre comidas?", ["no", "A veces", "Con frecuencia", "Siempre"])
        CH2O = st.slider("¿Cuánta agua bebes al día? (Litros)", 1.0, 3.0, 2.0)
        SCC = st.selectbox("¿Monitoreas tus calorías diariamente?", ["si", "no"])
        FAF = st.slider("¿Frecuencia de actividad física? (Días por semana)", 0.0, 3.0, 1.0)
        TUE = st.slider("¿Uso diario de dispositivos tecnológicos? (Horas)", 0.0, 2.0, 1.0)
        CALC = st.selectbox("¿Frecuencia de consumo de alcohol?", ["no", "A veces", "Con frecuencia", "Siempre"])
        MTRANS = st.selectbox("Medio de transporte habitual", ["transporte público", "automóvil", "a pie", "motocicleta", "bicicleta"])
        submit = st.form_submit_button("Calcular Nivel")
    
    if submit:
        input_data = pd.DataFrame([
            [edad, estatura, peso, f_con_sobrepeso, FAVC, FCVC, CAEC, CH2O, SCC, FAF, TUE, CALC, MTRANS]
        ], columns=["edad", "altura", "peso", "f_con_sobrepeso", "FAVC", "FCVC", "CAEC", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS"])
        
        # Convertiremos las columnas de Si/No a 1/0
        columnas_sino = ['f_con_sobrepeso', 'FAVC', 'SCC']
        for col in columnas_sino:
            input_data[col] = input_data[col].map({'si': 1, 'no': 0})
        # Convertiremos las columnas de A veces/Con frecuencia/Siempre a 0/1/2/3
        columnas_aveces = ['CALC', 'CAEC']
        for col in columnas_aveces:
            input_data[col] = input_data[col].map({"no" : 0, "A veces": 1, "Con frecuencia": 2, "Siempre": 3})
        # Convertiremos las columnas 
        columnas_transporte = ['MTRANS']
        for col in columnas_transporte:
            input_data[col] = input_data[col].map({"transporte público": 0, "automóvil": 1, "a pie": 2, "motocicleta": 3, "bicicleta": 4})
        
        modelo = joblib.load('prediccion_obesidad.pkl')
        scaler = joblib.load('mi_escalador_entrenado.pkl')
        cols_modelo = joblib.load('columnas_modelo.pkl')
        
        X_nuevos = input_data.copy()
        X_nuevos[cols_modelo] = scaler.transform(X_nuevos[cols_modelo])
        prediccion = modelo.predict(X_nuevos)
        mapeo = {
        0: 'Peso insuficiente',
        1: 'Peso normal',
        2: 'Sobrepeso nivel I',
        3: 'Sobrepeso nivel II',
        4: 'Obesidad tipo I',
        5: 'Obesidad tipo II',
        6: 'Obesidad tipo III'
        }
        prediccion_etiquetada = mapeo[prediccion[0]]
        st.write(f"El nivel de obesidad es: {prediccion_etiquetada}")
        st.image("Streamlit/logo.png")

# --- SECCIÓN 7: APPLICACIÓN AL MUNDO REAL ---
if seccion == "Aplicación al mundo real":
    st.title("Aplicación al mundo real")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Se puede aplicar este modelo para que las personas tomen conciencia de su nivel de obesidad, asi puedan mejorar su salud "   
            "y prevenir enfermedades relacionadas con el sobrepeso.")
        st.image("Streamlit/sillon.png")
    with col2:
        st.write("En colegios, institutos y universidades para que los jóvenes tomen conciencia de su nivel de obesidad y mejorar sus hábitos "
                    "alimenticios y de condición física.")
        st.image("Streamlit/joven.jpg")

# --- SECCIÓN 8: DESAFIOS Y APRENDIZAJES ---
if seccion == "Desafios y aprendizajes":
    st.title("Desafios y aprendizajes")
    
    col1, col2 = st.columns(2)
    
    st.image("Streamlit/imagen.png")
    st.divider()
    
    with col1:
        st.write("Desafios")
        st.write("- La categorización en 7 niveles (en lugar de solo 3) permite identificar perfiles específicos, lo que sugiere que los tratamientos deben ser segmentado y personalizados, no una solución única, estandarizada.")
        st.write("- Realizar la predicción antes, durante y después del tratamiento o si realiza cambios en su hábitos de vida.")
        st.write("- Implementar pausas activas o bailes entretenidos en colegios, institutos, universidades, empresas y en el hogar.")
    
    with col2:
        st.write("Aprendizajes")
        st.write("- Se demuestra que el Índice de Masa Corporal por sí solo no es suficiente. Factores como la actividad física y el consumo de alimentos entre comidas son predictores mucho más potentes para diferenciar entre los grados altos de obesidad.")
        st.write("- Se ha aprendido que modelos como Random Forest o Gadient suelen superar a la Regresión Logística en este caso, ya que las relaciones entre los hábitos de vida y el peso no son lineales.")
        st.write("- No sirve de nada que el modelo sea muy preciso, este no debe fallar en detectar correctamente a los pacientes con Obesidad Grado III, que son los de mayor riesgo.")
    
    
# --- SECCIÓN 9: MOTIVACIÓN ---
if seccion == "Motivación":
    st.title("Motivación")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("Streamlit/logo.png")
        st.write("Incentivar a las personas que tomen conciencia de su nivel de obesidad, mejorando sus hábitos alimenticios.")
    with col2:
        st.image("Streamlit/carrera.png")
        st.write("Crear intancias en familia, para potenciar la actividad física.")
    
    
