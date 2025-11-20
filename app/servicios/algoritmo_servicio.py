from modelo.ajustes import aplicar_ajuste
from modelo.agrupamiento import ejecutar_kmedias, ejecutar_kmodas
from modelo.chi_merge import chi_merge
from modelo.arbol_decision import entrenar_arbol
import pandas as pd
from pandas.api.types import is_numeric_dtype


def ejecutar_algoritmo(algoritmo, df, tipo_ajuste, parametros):
    if algoritmo == "kmedias":
        return _ejecutar_kmedias(df, tipo_ajuste, parametros)
    
    elif algoritmo == "kmodas":
        return _ejecutar_kmodas(df, parametros)
    
    elif algoritmo == "chiagrup":
        return _ejecutar_chi_agrupamiento(df, parametros)
    
    elif algoritmo == "arbol":
        return _ejecutar_arbol_decision(df, tipo_ajuste, parametros)
    
    elif algoritmo == "soloajuste":
        return _ejecutar_solo_ajuste(df, tipo_ajuste)
    
    else:
        raise ValueError(f"Algoritmo '{algoritmo}' no reconocido. Revisa el formulario.")

def _ejecutar_kmedias(df, tipo_ajuste, parametros):
    k = parametros.get("k")
    col_grupo  = parametros.get("col_grupo")
    col_valor  = parametros.get("col_valor")

    if not col_valor or not col_grupo:
        raise ValueError("Debes indicar la columna de valor numérico y la columna de clase para K-Medias.")
    
    return ejecutar_kmedias(df, tipo_ajuste, k)

def _ejecutar_kmodas(df, parametros):
    k = parametros.get("k")
    col_valor = (parametros.get("col_valor") or "").strip()
    col_grupo = parametros.get("col_grupo")
    if not col_valor or not col_grupo:
        raise ValueError("Debes indicar la columna categórica y la columna de clase para K-Modas.")
    
    return ejecutar_kmodas(df, k) 

def _ejecutar_chi_agrupamiento(df, parametros):
    columnas_numericas = (parametros.get("chi_col_atributo") or "").strip()
    columnas_categoricas = (parametros.get("chi_col_clase") or "").strip()
    
    if columnas_numericas not in df.columns:
        raise ValueError(
            f"La columna atributo '{columnas_numericas}' no existe en el archivo CSV."
        )
    if columnas_categoricas not in df.columns:
        raise ValueError(
            f"La columna de clase '{columnas_categoricas}' no existe en el archivo CSV."
        )

    if not columnas_numericas:
        raise ValueError("Se necesita al menos una columna numérica.")

    if not is_numeric_dtype(df[columnas_numericas]):
        try:
            df[columnas_numericas] = pd.to_numeric(df[columnas_numericas], errors="raise")
        except Exception:
            raise ValueError(
                f"La columna atributo '{columnas_numericas}' debe ser numérica o convertible a numérica."
            )

    chi_umbral_raw = parametros.get("chi_umbral")
    chi_umbral = float(chi_umbral_raw) if chi_umbral_raw not in (None, "",) else None

    df_resultado, intervalos, umbral_usado, _, _ = chi_merge(
        df, columnas_numericas, columnas_categoricas, chi_umbral
    )
    return {
        'df_resultado': df_resultado,
        'intervalos': intervalos,
        'umbral': umbral_usado,
        'col_numerica': columnas_numericas,
        'col_categorica': columnas_categoricas
    }


def _ejecutar_arbol_decision(df, tipo_ajuste, parametros):
    columna_objetivo = parametros.get("col_objetivo") or df.columns[-1]
    if not columna_objetivo:
        raise ValueError(
            "Debe indicarse la columna objetivo que el árbol debe predecir."
        )
    prueba = parametros.get("prueba") or 0.3
    profundidad_maxima = parametros.get("profundidad_max") or 4
    criterio = parametros.get("criterio", "gini")
    
    return entrenar_arbol(
        df, 
        columna_objetivo, 
        tipo_ajuste, 
        criterio,
        prueba, 
        profundidad_maxima
    )

def _ejecutar_solo_ajuste(df, tipo_ajuste):
    df_numerico = df.select_dtypes(include="number")
    df_no_numerico = df.select_dtypes(exclude="number")
    
    if df_numerico.empty:
        raise ValueError("No hay columnas numéricas para ajustar.")
    
    df_numerico_ajustado = aplicar_ajuste(df_numerico, tipo_ajuste)
    df_ajustado = pd.concat([df_numerico_ajustado, df_no_numerico], axis=1)
    
    return {
        'df_ajustado': df_ajustado,
        'tipo_ajuste': tipo_ajuste
    }
