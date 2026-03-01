import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

def arbol(df, col, n):
    X = df.drop(col, axis=1)
    y = df[col]
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Entrenamiento
    tree = DecisionTreeClassifier(max_depth=n, random_state=0)
    tree.fit(X_train, y_train)
    
    # Predicción
    pred = tree.predict(X_test)
    
    print("Accuracy:")
    accuracy = accuracy_score(y_test, pred) # accuracy
    print(accuracy)
    # precision
    print("Precision:")
    print(precision_score(y_test,pred, average='macro'))
    
    # recall
    print("Recall:")
    print(recall_score(y_test,pred, average='macro'))
    
    # F1-score
    print("F1-score:")
    print(f1_score(y_test, pred, average='macro'))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    # Importancia de variables
    tree_importance = {feature: importance for feature, importance in zip(X.columns, tree.feature_importances_)}
    return    

def escalado(df,col):
    # estudio de la variable a predecir
    X = df.drop(col, axis=1)
    y = df[col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    cols_a_escalar = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    scaler = StandardScaler()
    X_train_sc = X_train.copy()
    X_test_sc = X_test.copy()
    
    X_train_sc[cols_a_escalar] = scaler.fit_transform(X_train[cols_a_escalar])
    X_test_sc[cols_a_escalar] = scaler.transform(X_test[cols_a_escalar])
    
    X_train_sc = pd.DataFrame(X_train_sc, columns = X_train.columns)
    X_test_sc = pd.DataFrame(X_test_sc, columns = X_test.columns)
    
    return X_train_sc, X_test_sc, y_train, y_test

def dibujo(df, col,n):
    X = df.drop(col, axis=1)
    y = df[col]
    
    tree = DecisionTreeClassifier(max_depth=n, random_state=0)
    tree.fit(X, y)
    
    dot_data = export_graphviz(tree, out_file=None, 
                        filled=True, 
                        rounded=True, 
                        feature_names=X.columns,  
                        class_names=[str(c) for c in tree.classes_]) 
    return graphviz.Source(dot_data).render("arbol", format="png",view=True)

def baggin(df,col,n,m):
    # estudio de la variable a predecir
    X = df.drop(col, axis=1)
    y = df[col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    bagging_cla = BaggingClassifier(DecisionTreeClassifier(max_depth=20), 
                            n_estimators=n, # número arboles
                            max_samples = m) # número filas data set para entrenar cada modelo
    bagging_cla.fit(X_train, y_train)
    
    pred = bagging_cla.predict(X_test)
    
    print("Accuracy:")
    accuracy = accuracy_score(y_test, pred) # accuracy
    print(accuracy)
    # precision
    print("Precision:")
    print(precision_score(y_test,pred, average='macro'))
    
    # recall
    print("Recall:")
    print(recall_score(y_test,pred, average='macro'))
    
    # F1-score
    print("F1-score:")
    print(f1_score(y_test, pred, average='macro'))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

def pasting(df,col,n,m):
    # estudio de la variable a predecir
    X = df.drop(col, axis=1)
    y = df[col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    bagging_cla = BaggingClassifier(DecisionTreeClassifier(max_depth=20), 
                            n_estimators=n, # número arboles
                            max_samples = m, bootstrap = False) # número filas data set para entrenar cada modelo
    bagging_cla.fit(X_train, y_train)
    
    pred = bagging_cla.predict(X_test)
    
    print("Accuracy:")
    accuracy = accuracy_score(y_test, pred) # accuracy
    print(accuracy)
    # precision
    print("Precision:")
    print(precision_score(y_test,pred, average='macro'))
    
    # recall
    print("Recall:")
    print(recall_score(y_test,pred, average='macro'))
    
    # F1-score
    print("F1-score:")
    print(f1_score(y_test, pred, average='macro'))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

def random_forest(df,col,n,m):
    # estudio de la variable a predecir
    X = df.drop(col, axis=1)
    y = df[col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    forest = RandomForestClassifier(n_estimators=n,
                            max_depth=m)
    forest.fit(X_train, y_train)
    
    pred = forest.predict(X_test)
    
    print("Accuracy:")
    accuracy = accuracy_score(y_test, pred) # accuracy
    print(accuracy)
    # precision
    print("Precision:")
    print(precision_score(y_test,pred, average='macro'))
    
    # recall
    print("Recall:")
    print(recall_score(y_test,pred, average='macro'))
    
    # F1-score
    print("F1-score:")
    print(f1_score(y_test, pred, average='macro'))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

def adaBoost(df,col,n,m):
    # estudio de la variable a predecir
    X = df.drop(col, axis=1)
    y = df[col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    scaler = StandardScaler()
    
    # Identificar columnas numéricas (las que no son dummies de 0 y 1)
    cols_a_escalar = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cols_a_escalar] = scaler.fit_transform(X_train[cols_a_escalar])
    X_test_scaled[cols_a_escalar] = scaler.transform(X_test[cols_a_escalar])
    
    ada_cl = AdaBoostClassifier(DecisionTreeClassifier(max_depth=n),
                            n_estimators=m)
    ada_cl.fit(X_train_scaled, y_train)
    
    pred = ada_cl.predict(X_test_scaled)
    
    print("Accuracy:")
    accuracy = accuracy_score(y_test, pred) # accuracy
    print(accuracy)
    # precision
    print("Precision:")
    print(precision_score(y_test,pred, average='macro'))
    
    # recall
    print("Recall:")
    print(recall_score(y_test,pred, average='macro'))
    
    # F1-score
    print("F1-score:")
    print(f1_score(y_test, pred, average='macro'))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

def gradient_boost(df,col,n,m):
    # estudio de la variable a predecir
    X = df.drop(col, axis=1)
    y = df[col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    scaler = StandardScaler()
    
    # Identificar columnas numéricas (las que no son dummies de 0 y 1)
    cols_a_escalar = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cols_a_escalar] = scaler.fit_transform(X_train[cols_a_escalar])
    X_test_scaled[cols_a_escalar] = scaler.transform(X_test[cols_a_escalar])
    
    gb_cla= GradientBoostingClassifier(max_depth=n,
                                n_estimators=m)
    gb_cla.fit(X_train_scaled, y_train)
    
    pred = gb_cla.predict(X_test_scaled)
    
    print("Accuracy:")
    accuracy = accuracy_score(y_test, pred) # accuracy
    print(accuracy)
    # precision
    print("Precision:")
    print(precision_score(y_test,pred, average='macro'))
    
    # recall
    print("Recall:")
    print(recall_score(y_test,pred, average='macro'))
    
    # F1-score
    print("F1-score:")
    print(f1_score(y_test, pred, average='macro'))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

def logi_reg(df,col):
    # estudio de la variable a predecir
    X = df.drop(col, axis=1)
    y = df[col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    scaler = StandardScaler()
    
    # Identificar columnas numéricas (las que no son dummies de 0 y 1)
    cols_a_escalar = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cols_a_escalar] = scaler.fit_transform(X_train[cols_a_escalar])
    X_test_scaled[cols_a_escalar] = scaler.transform(X_test[cols_a_escalar])    
    
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    pred = model.predict(X_test_scaled)
    
    print("Accuracy:")
    accuracy = accuracy_score(y_test, pred)
    print(accuracy)
    # precision
    print("Precision:")
    print(precision_score(y_test,pred, average='macro'))
    
    # recall
    print("Recall:")
    print(recall_score(y_test,pred, average='macro'))
    
    # F1-score
    print("F1-score:")
    print(f1_score(y_test, pred, average='macro'))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

def knn(df,col,n):
    # estudio de la variable a predecir
    X = df.drop(col, axis=1)
    y = df[col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    print("Accuracy:")
    
    accuracy = accuracy_score(y_test, y_pred) # accuracy
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

def estandar_knn(df, col, n):
    # 1. Separar Features y Target
    X = df.drop(col, axis=1)
    y = df[col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    # 4. ESCALADO (Vital para KNN)
    # Solo escalamos las numéricas para no arruinar las dummies (0/1)
    scaler = StandardScaler()
    
    # Identificar columnas numéricas (las que no son dummies de 0 y 1)
    cols_a_escalar = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cols_a_escalar] = scaler.fit_transform(X_train[cols_a_escalar])
    X_test_scaled[cols_a_escalar] = scaler.transform(X_test[cols_a_escalar])
    
    # 5. Entrenamiento
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X_train_scaled, y_train)
    
    # 6. Predicciones y Evaluación
    y_pred = model.predict(X_test_scaled)
    
    print(f"--- Resultados KNN (k={n}) ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def optimizar_arbol(df, col, n):
    # Preparación de datos
    X = df.drop(col, axis=1)
    y = df[col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Definir el espacio de búsqueda de parámetros
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [6, 10, 23, 41],
        'min_samples_split': [1, 2, 3],
        'min_samples_leaf': [1, 2],
        'class_weight': [None, 'balanced'] # Útil para las 7 categorías si están desbalanceadas
    }

    # Configurar la búsqueda aleatoria
    search = GridSearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid=param_grid,
        cv=n,
        scoring='f1_macro',
        n_jobs=-1 # Usa todos los núcleos de tu procesador para ir más rápido
    )

    # Ajustar el modelo
    print(f"Buscando con GridSearchCV...")
    search.fit(X_train, y_train)
    
    # Evaluación con el mejor modelo encontrado
    best_tree = search.best_estimator_
    pred = best_tree.predict(X_test)

    # Reporte de resultados
    print("\n" + "="*30)
    print("MEJORES PARÁMETROS ENCONTRADOS:")
    print(search.best_params_)
    print("="*30)
    
    print("\nREPORTE DE CLASIFICACIÓN (7 Categorías):")
    # El classification_report es mejor para multiclase porque da detalle por categoría
    print(classification_report(y_test, pred))
    
    print("MATRIZ DE CONFUSIÓN:")
    print(confusion_matrix(y_test, pred))
    
    print("Accuracy:")
    accuracy = accuracy_score(y_test, pred) # accuracy
    print(accuracy)
    # precision
    print("Precision:")
    print(precision_score(y_test,pred, average='macro'))
    
    # recall
    print("Recall:")
    print(recall_score(y_test,pred, average='macro'))
    
    # F1-score
    print("F1-score:")
    print(f1_score(y_test, pred, average='macro'))

def randomizar_arbol(df, col, n):
    # Preparación de datos
    X = df.drop(col, axis=1)
    y = df[col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Definir el espacio de búsqueda de parámetros
    param_dist = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [int(x) for x in np.linspace(start = 10, stop = 50, num = 10)],
        'min_samples_split': [int(x) for x in np.linspace(start = 2, stop = 10, num = 10)],
        'min_samples_leaf': [int(x) for x in np.linspace(start = 1, stop = 10, num = 10)],
        'class_weight': [None, 'balanced'] # Útil para las 7 categorías si están desbalanceadas
    }

    # Configurar la búsqueda aleatoria
    search = RandomizedSearchCV(
        estimator=DecisionTreeClassifier(random_state=0),
        param_distributions=param_dist,
        n_iter=n,
        cv=5,
        scoring='f1_macro',
        random_state=0,
        n_jobs=-1 # Usa todos los núcleos de tu procesador para ir más rápido
    )

    # Ajustar el modelo
    print(f"Buscando los mejores parámetroscon {n} combinaciones...")
    search.fit(X_train, y_train)
    
    # Evaluación con el mejor modelo encontrado
    best_tree = search.best_estimator_
    pred = best_tree.predict(X_test)

    # Reporte de resultados
    print("\n" + "="*30)
    print("MEJORES PARÁMETROS ENCONTRADOS:")
    print(search.best_params_)
    print("="*30)
    
    print("\nREPORTE DE CLASIFICACIÓN (7 Categorías):")
    # El classification_report es mejor para multiclase porque da detalle por categoría
    print(classification_report(y_test, pred))
    
    print("MATRIZ DE CONFUSIÓN:")
    print(confusion_matrix(y_test, pred))
    
    print("Accuracy:")
    accuracy = accuracy_score(y_test, pred) # accuracy
    print(accuracy)
    # precision
    print("Precision:")
    print(precision_score(y_test,pred, average='macro'))
    
    # recall
    print("Recall:")
    print(recall_score(y_test,pred, average='macro'))
    
    # F1-score
    print("F1-score:")
    print(f1_score(y_test, pred, average='macro'))

def optimizar_forest(df, col_objetivo, n):
    # Preparación de datos
    X = df.drop(col_objetivo, axis=1)
    y = df[col_objetivo]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Espacio de búsqueda específico para Random Forest
    param_grid = {
        'n_estimators': [32, 40, 45, 50], # Número de árboles en el bosque
        'max_depth': [1500, 1888, 2000, 3000],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False],           # Método de selección de muestras
        'max_features': ['sqrt', 'log2']      # Cantidad de variables por árbol
    }
    
    # RandomForest es más pesado, n_jobs=-1 es fundamental aquí
    search = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid,
        cv=n,                
        scoring='f1_macro',   # Ideal para 7 categorías
        n_jobs=-1
    )

    # Ajustar el modelo
    print(f"Entrenando con GridSearchCV...")
    search.fit(X_train, y_train)
    
    # Evaluación
    best_forest = search.best_estimator_
    pred = best_forest.predict(X_test)

    # Reporte de resultados
    print("\n" + "="*35)
    print("MEJORES PARÁMETROS:")
    print(search.best_params_)
    print("="*35)
    
    print("\nREPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, pred))
    
    print("MATRIZ DE CONFUSIÓN:")
    print(confusion_matrix(y_test, pred))

    print("Accuracy:")
    accuracy = accuracy_score(y_test, pred) # accuracy
    print(accuracy)
    # precision
    print("Precision:")
    print(precision_score(y_test,pred, average='macro'))
    
    # recall
    print("Recall:")
    print(recall_score(y_test,pred, average='macro'))
    
    # F1-score
    print("F1-score:")
    print(f1_score(y_test, pred, average='macro'))

def randomizar_forest(df, col_objetivo,n):
    # Preparación de datos
    X = df.drop(col_objetivo, axis=1)
    y = df[col_objetivo]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Espacio de búsqueda específico para Random Forest
    param_dist = {
        'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 50, num = 10)], # Número de árboles en el bosque
        'max_depth': [int(x) for x in np.linspace(start = 500, stop = 3000, num = 10)],
        'min_samples_split': [int(x) for x in np.linspace(start = 1, stop = 10, num = 10)],
        'min_samples_leaf': [int(x) for x in np.linspace(start = 1, stop = 10, num = 10)],
        'bootstrap': [True, False],           # Método de selección de muestras
        'max_features': ['sqrt', 'log2']      # Cantidad de variables por árbol
    }

    # RandomForest es más pesado, n_jobs=-1 es fundamental aquí
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=0),
        param_distributions=param_dist,
        n_iter=n,
        cv=3,                 # Reducimos a 3 folds para que sea más rápido
        scoring='f1_macro',   # Ideal para 7 categorías
        random_state=0,
        n_jobs=-1
    )

    # Ajustar el modelo
    print(f"Entrenando Random Forest con {n} combinaciones...")
    search.fit(X_train, y_train)
    
    # Evaluación
    best_forest = search.best_estimator_
    pred = best_forest.predict(X_test)

    # Reporte de resultados
    print("\n" + "="*35)
    print("MEJORES PARÁMETROS RANDOM FOREST:")
    print(search.best_params_)
    print("="*35)
    
    print("\nREPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, pred))
    
    print("MATRIZ DE CONFUSIÓN:")
    print(confusion_matrix(y_test, pred))

    print("Accuracy:")
    accuracy = accuracy_score(y_test, pred) # accuracy
    print(accuracy)
    # precision
    print("Precision:")
    print(precision_score(y_test,pred, average='macro'))
    
    # recall
    print("Recall:")
    print(recall_score(y_test,pred, average='macro'))
    
    # F1-score
    print("F1-score:")
    print(f1_score(y_test, pred, average='macro'))

def optimizar_gradient(df,col,n):
    # estudio de la variable a predecir
    X = df.drop(col, axis=1)
    y = df[col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    scaler = StandardScaler()
    
    # Identificar columnas numéricas (las que no son dummies de 0 y 1)
    cols_a_escalar = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cols_a_escalar] = scaler.fit_transform(X_train[cols_a_escalar])
    X_test_scaled[cols_a_escalar] = scaler.transform(X_test[cols_a_escalar])
    
    # Definir el "Grid" de parámetros a probar
    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    # Configurar la búsqueda
    gb = GradientBoostingClassifier()
    grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, 
                            cv=n, scoring='accuracy', n_jobs=-1)
    
    # Ejecutar la optimización
    grid_search.fit(X_train_scaled, y_train)
    
    # El mejor modelo encontrado
    best_model = grid_search.best_estimator_
    
    # predicción
    pred = best_model.predict(X_test_scaled)
    # Reporte de resultados
    print("\n" + "="*35)
    print("MEJORES PARÁMETROS:")
    print(best_model)
    print("="*35)
    
    print("\nREPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, pred))
    
    print("MATRIZ DE CONFUSIÓN:")
    print(confusion_matrix(y_test, pred))
    print("Accuracy:")
    accuracy = accuracy_score(y_test, pred) # accuracy
    print(accuracy)
    # precision
    print("Precision:")
    print(precision_score(y_test,pred, average='macro'))
    
    # recall
    print("Recall:")
    print(recall_score(y_test,pred, average='macro'))
    
    # F1-score
    print("F1-score:")
    print(f1_score(y_test, pred, average='macro'))


def randomizar_gradient(df,col,n):
    # estudio de la variable a predecir
    X = df.drop(col, axis=1)
    y = df[col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    scaler = StandardScaler()
    
    # Identificar columnas numéricas (las que no son dummies de 0 y 1)
    cols_a_escalar = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cols_a_escalar] = scaler.fit_transform(X_train[cols_a_escalar])
    X_test_scaled[cols_a_escalar] = scaler.transform(X_test[cols_a_escalar])
    
    # Definir el "Random" de parámetros a probar
    param_dist = {
        'max_depth': [int(x) for x in np.linspace(start = 500, stop = 3000, num = 10)],
        'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 50, num = 10)],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    # Configurar la búsqueda
    gb = GradientBoostingClassifier()
    grid_search = RandomizedSearchCV(estimator=gb, param_distributions=param_dist, 
                            cv=n, scoring='accuracy', n_jobs=-1)
    
    # Ejecutar la optimización
    grid_search.fit(X_train_scaled, y_train)
    
    # El mejor modelo encontrado
    best_model = grid_search.best_estimator_
    
    # predicción
    pred = best_model.predict(X_test_scaled)
    # Reporte de resultados
    print("\n" + "="*35)
    print("MEJORES PARÁMETROS:")
    print(best_model)
    print("="*35)
    
    print("\nREPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, pred))
    
    print("MATRIZ DE CONFUSIÓN:")
    print(confusion_matrix(y_test, pred))
    print("Accuracy:")
    accuracy = accuracy_score(y_test, pred) # accuracy
    print(accuracy)
    # precision
    print("Precision:")
    print(precision_score(y_test,pred, average='macro'))
    
    # recall
    print("Recall:")
    print(recall_score(y_test,pred, average='macro'))
    
    # F1-score
    print("F1-score:")
    print(f1_score(y_test, pred, average='macro'))
