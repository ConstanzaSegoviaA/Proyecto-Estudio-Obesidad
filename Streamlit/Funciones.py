import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def explorar_df(df):
    print(df.info())

    print('Primeras filas')
    print(df.head())

    print('Describe()')
    print(df.describe(include='all').T)

    print('Nulos')
    nulos = df.isnull().sum()
    print(nulos[nulos > 0] if nulos.any() else 'Notnnull')

    print('Duplicados')
    print(df.duplicated().sum())
    
    print('Tamaño')
    print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")

def limpiar_dataset(df):
    df_clean = df.copy()
    df_clean.columns = (df_clean.columns
                        .str.strip()
                        .str.lower()
                        .str.replace(' ', '_')
                        .str.replace('.', '', regex=False))
    df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(how='all') 
    return df_clean

def categorico(df,col):
    """Realiza un análisis descriptivo de una columna categórica."""
    print(df[col].value_counts())
    print(df[col].unique())
    print(df[col].nunique())

def numerico(df,col):
    """Realiza un análisis descriptivo de una columna numérica."""
    print(df[col].describe())
    print(df[col].isnull().sum())
    print(df[col].nunique())

def filtrar_fila(df,col,lista):
    return df[df[col].isin(lista)]

def completar_nulos(df,col,valor):
    df[col] = df[col].fillna(valor)
    return df

def estadisticos(df):
    print(df.describe())
    print(df.select_dtypes(include=["number"]).describe())

def ver_nulos(df):
    df_con_nulos = df[df.isnull().any(axis=1)]
    display(df_con_nulos)

def ver_duplicados(df):
    df_duplicados = df[df.duplicated()]
    display(df_duplicados)
    return df_duplicados

def asignar_secciones(df):
    mapeo= {'Normal_Weight':'Peso normal', 'Overweight_Level_I': 'Sobrepeso nivel I',
            'Overweight_Level_II': 'Sobrepeso nivel II','Obesity_Type_I': 'Obesidad tipo I', 'Insufficient_Weight': 'Peso insuficiente', 'Obesity_Type_II': 'Obesidad tipo II',
            'Obesity_Type_III': 'Obesidad tipo III'}
    df['nivel_obesidad'] = df['nivel_obesidad'].map(mapeo)
    return df

def cambio(df,col):
    valores= {'Sometimes':'A veces', 'Frequently':'Con frecuencia', 'Always':'Siempre',"no":"no"}
    df[col] = df[col].map(valores)
    return df

def si_no(df,col):
    afirmacion= {'yes':'Si', 'no':'No'}
    df[col] = df[col].map(afirmacion)
    return df

def se(df,col):
    num={"Male":"masculino","Female":"femenino"}
    df[col] = df[col].map(num)
    return df

def traspaso(df,col):
    sig={"Public_Transportation": "transporte público","Automobile":"automóvil","Walking": "a pie","Motorbike":"motocicleta","Bike":"bicicleta"}
    df[col] = df[col].map(sig)
    return df

def traspaso_estudio(df):
    df_estudio=df.copy()
    # Convertiremos el género a binario (1 para masculino, 0 para femenino por ejemplo)
    df_estudio['género'] = df_estudio['género'].map({'femenino': 0, 'masculino': 1})

    # Convertiremos las columnas de Si/No a 1/0
    columnas_sino = ['f_con_sobrepeso', 'FAVC', 'fuma', 'SCC']
    for col in columnas_sino:
        df_estudio[col] = df_estudio[col].map({'Si': 1, 'No': 0})

    # Convertiremos las columnas de A veces/Con frecuencia/Siempre a 0/1/2/3
    columnas_aveces = ['CALC', 'CAEC']
    for col in columnas_aveces:
        df_estudio[col] = df_estudio[col].map({'no': 0, 'A veces': 1, 'Con frecuencia': 2, 'Siempre': 3})

    # Convertiremos las columnas de Public_Transportation/Automobile/Walking/Motorbike/Bike a 0/1/2/3/4
    columnas_transporte = ['MTRANS']
    for col in columnas_transporte:
        df_estudio[col] = df_estudio[col].map({'transporte público': 0, 'automóvil': 1, 'a pie': 2, 'motocicleta': 3, 'bicicleta': 4})

    # Convertiremos las columnas
    columnas_obesidad = ['nivel_obesidad']
    for col in columnas_obesidad:
        df_estudio[col] = df_estudio[col].map({ 'Peso insuficiente': 0,'Peso normal': 1, 'Sobrepeso nivel I': 2, 'Sobrepeso nivel II': 3, 'Obesidad tipo I': 4, 'Obesidad tipo II': 5, 'Obesidad tipo III': 6})
    return df_estudio

def corr_heatmap(df):
    corr=np.abs(df.corr())
    # Set up mask for triangle representation
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 10))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask,  vmax=1,square=True, linewidths=.5, cbar_kws={"shrink": .5},annot = corr)
    return plt.show()

def cambiar(df):
    # Convertiremos las columnas de A veces/Con frecuencia/Siempre a 0/1/2/3
    columnas_aveces = ['CALC', 'CAEC']
    for col in columnas_aveces:
        df[col] = df[col].map({'no': 0, 'A veces': 1, 'Con frecuencia': 2, 'Siempre': 3})
    # Convertiremos las columnas
    columnas_obesidad = ['nivel_obesidad']
    for col in columnas_obesidad:
        df[col] = df[col].map({ 'Peso insuficiente': 0,'Peso normal': 1, 'Sobrepeso nivel I': 2, 'Sobrepeso nivel II': 3, 'Obesidad tipo I': 4, 'Obesidad tipo II': 5, 'Obesidad tipo III': 6})
    return df

def volver(df, col):
    mapeo = {
        0: 'Peso insuficiente',
        1: 'Peso normal',
        2: 'Sobrepeso nivel I',
        3: 'Sobrepeso nivel II',
        4: 'Obesidad tipo I',
        5: 'Obesidad tipo II',
        6: 'Obesidad tipo III'
    }
    
    df[col] = df[col].map(mapeo)
    return df