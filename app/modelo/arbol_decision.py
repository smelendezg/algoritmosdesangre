import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from .ajustes import aplicar_ajuste

def entrenar_arbol(df, col_objetivo, tipo_ajuste, criterio, prueba=0.3, profundidad_max=4):

    _validar_parametros_basicos(df, col_objetivo, criterio)

    filas_faltantes = df[col_objetivo].isna()

    if filas_faltantes.any():
        return _entrenar_con_faltantes(
            df, col_objetivo, tipo_ajuste, criterio, profundidad_max
        )
    else:
        return _entrenar_sin_faltantes(
            df, col_objetivo, tipo_ajuste, criterio, prueba, profundidad_max
        )

def _validar_parametros_basicos(df, col_objetivo, criterio):
    if col_objetivo not in df.columns:
        raise ValueError(f"La columna objetivo '{col_objetivo}' no existe en el CSV.")

    if criterio not in ['gini', 'entropy']:
        raise ValueError(
            f"Criterio '{criterio}' no válido. Use 'gini' o 'entropy'."
        )
    

def _entrenar_con_faltantes(df, col_objetivo, tipo_ajuste, criterio, profundidad_max):

    filas_faltantes = df[col_objetivo].isna()
    df_entrenamiento = df[~filas_faltantes].copy()
    df_prueba = df[filas_faltantes].copy()

    if df_entrenamiento.empty:
        raise ValueError(
            f"No hay filas con la clase '{col_objetivo}' conocida para entrenar el árbol."
        )
    
    df_tmp = pd.concat([df_entrenamiento, df_prueba], axis=0)
    X_todas, y_todas = _preparar_datos(df_tmp, col_objetivo, tipo_ajuste)

    X_entrenamiento = X_todas.loc[df_entrenamiento.index]
    y_entrenamiento = y_todas.loc[df_entrenamiento.index]

    valid = ~X_entrenamiento.isna().any(axis=1)
    X_entrenamiento = X_entrenamiento[valid]
    y_entrenamiento = y_entrenamiento[valid]

    arbol = DecisionTreeClassifier(criterion=criterio, max_depth=profundidad_max, random_state=0)
    arbol.fit(X_entrenamiento, y_entrenamiento)

    y_pred_train = arbol.predict(X_entrenamiento)
    exactitud = accuracy_score(y_entrenamiento, y_pred_train)

    df_resultados = X_entrenamiento.copy()
    df_resultados["real"] = y_entrenamiento.values
    df_resultados["prediccion"] = y_pred_train

    pred_faltantes = None
    if not df_prueba.empty:
        X_prueba = X_todas.loc[df_prueba.index]
        y_pred_faltantes = arbol.predict(X_prueba)

        df_pred = df_prueba.copy()
        df_pred[col_objetivo] = y_pred_faltantes
        pred_faltantes = df_pred

    reglas_texto = export_text(arbol, feature_names=list(X_todas.columns))
    profundidad = arbol.get_depth()
    n_hojas = arbol.get_n_leaves()

    return {
        'modelo': arbol,
        'exactitud': exactitud,          
        'predicciones': df_resultados,
        'tipo_ajuste': tipo_ajuste,
        'criterio': criterio,
        'reglas_texto': reglas_texto,
        'profundidad': profundidad,
        'n_hojas': n_hojas,
        'pred_faltantes': pred_faltantes,
        'modo_eval': 'entrenamiento'
    }

def _entrenar_sin_faltantes(df, col_objetivo, tipo_ajuste, criterio, prueba, profundidad_max):
    _preparar_datos(df, col_objetivo, tipo_ajuste)

    valid = ~X.isna().any(axis=1)
    X = X[valid]
    y = y[valid]

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=prueba, random_state=42)

    arbol = DecisionTreeClassifier(criterion=criterio, max_depth=profundidad_max,random_state=0)
    arbol.fit(X_entrenamiento, y_entrenamiento)

    y_prediccion = arbol.predict(X_prueba)
    exactitud = accuracy_score(y_prueba, y_prediccion)

    df_resultados = X_prueba.copy()
    df_resultados["real"] = y_prueba.values
    df_resultados["prediccion"] = y_prediccion

    reglas_texto = export_text(arbol, feature_names=list(X.columns))
    profundidad = arbol.get_depth()
    n_hojas = arbol.get_n_leaves()

    return {
        'modelo': arbol,
        'exactitud': exactitud,         
        'predicciones': df_resultados,
        'tipo_ajuste': tipo_ajuste,
        'criterio': criterio,
        'reglas_texto': reglas_texto,
        'profundidad': profundidad,
        'n_hojas': n_hojas,
        'pred_faltantes': None,
        'modo_eval': 'validacion'
    }


def _preparar_datos(df, col_objetivo, tipo_ajuste):
    X_num = df.select_dtypes(include="number")
    X_no_num = df.select_dtypes(exclude="number")
    
    if not X_num.empty:
        X_num_ajustado = aplicar_ajuste(X_num, tipo_ajuste)
        X = pd.concat([X_num_ajustado, X_no_num], axis=1)
    else:
        X = df.copy()
    
    X = X.drop(columns=[col_objetivo], errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    y = df[col_objetivo]

    return X, y