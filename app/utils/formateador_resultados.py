import pandas as pd
from sklearn.tree import plot_tree
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def formatear_resultado(algoritmo, resultado, tipo_ajuste):
    formateadores = {
        'kmedias': formatear_kmedias,
        'kmodas': formatear_kmodas,
        'chiagrup': formatear_chiagrup,
        'arbol': formatear_arbol,
        'soloajuste': formatear_ajuste
    }
    
    formateador = formateadores.get(algoritmo)
    if not formateador:
        raise ValueError(f"No existe formateador para el algoritmo '{algoritmo}'")
    
    return formateador(resultado, tipo_ajuste)


def formatear_kmedias(resultado, tipo_ajuste):
    df_resultado = resultado['df']
    k = resultado['k']

    stats = df_resultado.groupby('grupo').size().reset_index(name='cantidad')
    
    return {
        'titulo': 'Resultados de K-Medias',
        'algoritmo': 'kmedias',
        'metricas': [
            {'label': 'Clusters creados', 'value': k, 'type': 'info'},
            {'label': 'Datos procesados', 'value': len(df_resultado), 'type': 'info'},
            {'label': 'Ajuste aplicado', 'value': tipo_ajuste, 'type': 'info'}
        ],
        'tabla_principal': df_resultado.head(50).to_html(classes='table', index=False),
        'tabla_stats': stats.to_html(classes='table', index=False),
        'info': f'Los datos han sido agrupados en {k} clusters usando K-Medias.'
    }


def formatear_kmodas(resultado, tipo_ajuste):
    df_resultado = resultado['df']
    k = resultado['k']
    
    stats = df_resultado.groupby('grupo').size().reset_index(name='cantidad')
    
    return {
        'titulo': 'Resultados de K-Modas',
        'algoritmo': 'kmodas',
        'metricas': [
            {'label': 'Clusters creados', 'value': k, 'type': 'info'},
            {'label': 'Datos procesados', 'value': len(df_resultado), 'type': 'info'}
        ],
        'tabla_principal': df_resultado.head(50).to_html(classes='table', index=False),
        'tabla_stats': stats.to_html(classes='table', index=False),
        'info': f'Los datos categóricos han sido agrupados en {k} clusters usando K-Modas.'
    }


def formatear_chiagrup(resultado, tipo_ajuste):
    df_resultado = resultado['df_resultado']
    intervalos = resultado['intervalos']
    umbral = resultado['umbral']
    col_numerica = resultado['col_numerica']
    col_categorica = resultado['col_categorica']
    
    return {
        'titulo': 'Resultados de Chi-Agrupamiento',
        'algoritmo': 'chiagrup',
        'metricas': [
            {'label': 'Intervalos creados', 'value': len(intervalos), 'type': 'info'},
            {'label': 'Umbral χ² usado', 'value': f'{umbral:.3f}', 'type': 'info'},
            {'label': 'Columna numérica', 'value': col_numerica, 'type': 'info'},
            {'label': 'Columna categórica', 'value': col_categorica, 'type': 'info'}
        ],
        'tabla_intervalos': _formatear_tabla_intervalos(intervalos),
        'tabla_principal': df_resultado.head(50).to_html(classes='table', index=False),
        'info': f'Se discretizó la columna "{col_numerica}" según la clase "{col_categorica}".'
    }


def formatear_arbol(resultado, tipo_ajuste):
    exactitud = resultado['exactitud']
    df_predicciones = resultado['predicciones']
    criterio = resultado['criterio']
    reglas_texto = resultado.get('reglas_texto', '')
    profundidad = resultado.get('profundidad', 'N/A')
    n_hojas = resultado.get('n_hojas', 'N/A')
    imagen_arbol = None
    modelo = resultado.get('modelo')
    if modelo is not None:
        try:
            # columnas que usó el modelo (quitamos real y prediccion)
            feature_cols = [
                c for c in df_predicciones.columns
                if c not in ('real', 'prediccion')
            ]

            fig, ax = plt.subplots(figsize=(14, 7))
            plot_tree(
                modelo,
                feature_names=feature_cols,
                class_names=[str(c) for c in modelo.classes_],
                filled=True,
                rounded=True,
                fontsize=8,
                ax=ax
            )

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            imagen_arbol = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            print("Error al generar imagen del árbol:", e)
            imagen_arbol = None
    
    correcto = (df_predicciones['real'] == df_predicciones['prediccion']).sum()
    incorrecto = len(df_predicciones) - correcto
    
    tabla_pred_faltantes = None
    if resultado.get('pred_faltantes') is not None:
        df_faltantes = resultado['pred_faltantes']
        tabla_pred_faltantes = df_faltantes.to_html(classes='table', index=False)

    return {
        'titulo': 'Resultados de Árbol de Decisión',
        'algoritmo': 'arbol',
        'metricas': [
            {'label': 'Exactitud', 'value': f'{exactitud:.1%}', 'type': 'success'},
            {'label': 'Predicciones correctas', 'value': correcto, 'type': 'success'},
            {'label': 'Predicciones incorrectas', 'value': incorrecto, 'type': 'info'},
            {'label': 'Profundidad del árbol', 'value': profundidad, 'type': 'info'},
            {'label': 'Número de hojas', 'value': n_hojas, 'type': 'info'},
            {'label': 'Criterio usado', 'value': criterio, 'type': 'info'},
            
        ],
        'reglas_texto': reglas_texto,
        'tabla_pred_faltantes': tabla_pred_faltantes,
        'info': f'Modelo entrenado con criterio {criterio} y ajuste {tipo_ajuste}.',
        'img_arbol': imagen_arbol
    }


def formatear_ajuste(resultado, tipo_ajuste):
    df_ajustado = resultado['df_ajustado']
    
    n_columnas_numericas = len(df_ajustado.select_dtypes(include='number').columns)
    
    return {
        'titulo': 'Datos Normalizados',
        'algoritmo': 'ajuste',
        'metricas': [
            {'label': 'Tipo de ajuste', 'value': tipo_ajuste, 'type': 'info'},
            {'label': 'Filas procesadas', 'value': len(df_ajustado), 'type': 'info'},
            {'label': 'Columnas numéricas', 'value': n_columnas_numericas, 'type': 'info'}
        ],
        'tabla_principal': df_ajustado.head(50).to_html(classes='table', index=False),
        'info': f'Se aplicó normalización tipo "{tipo_ajuste}" a las columnas numéricas.'
    }


def _formatear_tabla_intervalos(intervalos):
    filas = []
    for i, inter in enumerate(intervalos):
        filas.append({
            'Cluster': i,
            'Mínimo': inter['min'],
            'Máximo': inter['max'],
            'Total': int(inter['conteo'].sum())
        })
    
    df_intervalos = pd.DataFrame(filas)
    return df_intervalos.to_html(classes='table', index=False)