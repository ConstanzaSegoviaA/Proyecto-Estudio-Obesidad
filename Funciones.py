import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz


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

def estudio(df,col,n):
    # estudio de la variable a predecir
    X = df.drop(col, axis=1)
    y = df[col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    print("Accuracy:")
    
    accuracy = knn.score(X_test, y_test) # accuracy
    print(accuracy)
    # precision
    print("Precision:")
    print(precision_score(y_test,y_pred, average='macro'))
    
    # recall
    print("Recall:")
    print(recall_score(y_test,y_pred, average='macro'))
    
    # F1-score
    print("F1-score:")
    print(f1_score(y_test, y_pred, average='macro'))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

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

def arbol(df, col):
    X = df.drop(col, axis=1)
    y = df[col]
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Normalización
    normalizer = MinMaxScaler()
    X_train_norm = normalizer.fit_transform(X_train)
    X_test_norm = normalizer.transform(X_test)
    
    # Entrenamiento
    tree = DecisionTreeRegressor(max_depth=10, random_state=0)
    tree.fit(X_train_norm, y_train)
    
    # Predicción
    pred = tree.predict(X_test_norm)
    
    # Métricas (Corregido 'y_pred' por 'pred')
    print("MAE:", mean_absolute_error(y_test, pred))
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print("RMSE:", rmse)
    print("R2 score:", tree.score(X_test_norm, y_test))
    
    # Importancia de variables
    tree_importance = {feature: importance for feature, importance in zip(X.columns, tree.feature_importances_)}
    
    return tree, tree_importance   

def normalizar(df,col):
    # estudio de la variable a predecir
    X = df.drop(col, axis=1)
    y = df[col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    normalizer = MinMaxScaler()
    normalizer.fit(X_train)
    X_train_norm = normalizer.transform(X_train)
    X_test_norm = normalizer.transform(X_test)
    
    X_train_norm = pd.DataFrame(X_train_norm, columns = X_train.columns)
    X_test_norm = pd.DataFrame(X_test_norm, columns = X_test.columns)
    
    return X_train_norm, X_test_norm, y_train, y_test


def dibujo(df, col):
    X = df.drop(col, axis=1)
    y = df[col]
    
    tree = DecisionTreeClassifier(max_depth=5, random_state=0)
    tree.fit(X, y)
    
    dot_data = export_graphviz(tree, out_file=None, 
                        filled=True, 
                        rounded=True, 
                        feature_names=X.columns,  
                        class_names=[str(c) for c in tree.classes_]) 
    return graphviz.Source(dot_data)