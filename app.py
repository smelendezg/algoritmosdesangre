from flask import Flask, render_template, request 
import pandas as pd
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kmodes.kmodes import KModes
from scipy.stats import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

def k_por_defecto(n_filas):
    try:
        n = int(n_filas)
    except:
        n = 2
    return max(2, min(8, int(math.sqrt(max(n, 1)))))


def ajuste_lineal_df(df_num):
    df_out = df_num.astype(float).copy()
    for col in df_out.columns:
        x_min = df_out[col].min()
        x_max = df_out[col].max()
        if pd.isna(x_min) or pd.isna(x_max):
            df_out[col] = 0.0
        elif x_max == x_min:
            df_out[col] = 0.0
        else:
            df_out[col] = (df_out[col] - x_min) / (x_max - x_min)
    return df_out

def ajuste_zscore_df(df_num):
    df_out = df_num.astype(float).copy()
    for col in df_out.columns:
        mu = df_out[col].mean()
        sigma = df_out[col].std(ddof=0)
        if sigma == 0 or pd.isna(sigma):
            df_out[col] = df_out[col] - mu
        else:
            df_out[col] = (df_out[col] - mu) / sigma
    return df_out

def ajuste_log_df(df_num):
    df_out = df_num.astype(float).copy()
    for col in df_out.columns:
        minimo = df_out[col].min()
        if pd.isna(minimo):
            df_out[col] = 0.0
            continue
        if minimo <= 0:
            df_out[col] = df_out[col] - minimo + 1
        df_out[col] = np.log(df_out[col])
    return df_out

def aplicar_ajuste(df_num, tipo_ajuste):
    if df_num.empty:
        return df_num

    if tipo_ajuste == "lineal":
        return ajuste_lineal_df(df_num)
    elif tipo_ajuste == "zscore":
        return ajuste_zscore_df(df_num)
    elif tipo_ajuste == "algoritmico":
        return df_num.astype(float).copy()
    elif tipo_ajuste == "ninguno":
        return df_num.astype(float).copy()
    else:
        return df_num.astype(float).copy()


def chi_merge(df, attr_col, class_col, threshold=None, alpha=0.05):
    data = df[[attr_col, class_col]].dropna().copy()
    clases = sorted(data[class_col].unique())
    n_clases = len(clases)

    if n_clases == 0:
        raise ValueError("No hay clases para Chi-Merge.")

    if threshold is None:
        df_chi = n_clases - 1
        threshold = chi2.ppf(1 - alpha, df_chi)

    agrupado = data.groupby(attr_col)[class_col].value_counts().unstack(fill_value=0)
    agrupado = agrupado.reindex(columns=clases, fill_value=0).sort_index()

    intervalos = []
    for v, fila in agrupado.iterrows():
        intervalos.append({"min": v, "max": v, "counts": fila.values.astype(float)})

    def chi2_entre(i1, i2):
        obs = np.vstack([i1["counts"], i2["counts"]])
        fila_tot = obs.sum(axis=1, keepdims=True)
        col_tot = obs.sum(axis=0, keepdims=True)
        total = obs.sum()
        if total == 0:
            return 0.0
        esper = fila_tot @ col_tot / total
        esper[esper == 0] = 1e-9
        return ((obs - esper) ** 2 / esper).sum()

    fusion = True
    while fusion and len(intervalos) > 1:
        fusion = False
        chis = []
        for i in range(len(intervalos) - 1):
            try:
                c = chi2_entre(intervalos[i], intervalos[i+1])
            except:
                c = 0.0
            chis.append((c, i))

        chis.sort(key=lambda x: x[0])
        chi_min, idx = chis[0]

        if chi_min < threshold:
            a = intervalos[idx]
            b = intervalos[idx+1]
            nuevo = {
                "min": a["min"],
                "max": b["max"],
                "counts": a["counts"] + b["counts"],
            }
            intervalos[idx:idx+2] = [nuevo]
            fusion = True

    etiquetas = []
    etiquetas_intervalo = []
    for _, fila in df.iterrows():
        val = fila[attr_col]
        puesto = False
        for i, inter in enumerate(intervalos):
            if inter["min"] <= val <= inter["max"]:
                etiquetas.append(i)
                etiquetas_intervalo.append(f"[{inter['min']}, {inter['max']}]")
                puesto = True
                break
        if not puesto:
            etiquetas.append(-1)
            etiquetas_intervalo.append("sin_intervalo")

    df_res = df.copy()
    df_res["chi_cluster"] = etiquetas
    df_res["intervalo"] = etiquetas_intervalo

    filas_tabla = []
    for i, inter in enumerate(intervalos):
        fila = {
            "cluster": i,
            "min": inter["min"],
            "max": inter["max"],
            "total": int(inter["counts"].sum())
        }
        for cls, cnt in zip(clases, inter["counts"]):
            fila[str(cls)] = int(cnt)
        filas_tabla.append(fila)
    inter_df = pd.DataFrame(filas_tabla)

    return df_res, inter_df, threshold, attr_col, class_col


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run():
    try:
        archivo = request.files.get("file")
        if not archivo or archivo.filename == "":
            return "Error: no subiste archivo CSV "

        try:
            df = pd.read_csv(archivo)
        except Exception as e:
            return f"Error leyendo el CSV (¿es realmente un .csv?): {e}"

        algoritmo = request.form.get("algorithm")  
        tipo_ajuste = request.form.get("ajuste", "algoritmico")

        n_clusters_str = request.form.get("n_clusters")
        chi_threshold_str = request.form.get("chi_threshold")
        target_col = request.form.get("target_col")
        max_depth_str = request.form.get("max_depth")
        test_size_str = request.form.get("test_size")

        if n_clusters_str:
            try:
                k = int(n_clusters_str)
            except ValueError:
                return "K no es un número entero válido."
            if k < 2:
                return "K debe ser al menos 2."
        else:
            k = k_por_defecto(len(df))

        if algoritmo == "kmedias":
            num_df = df.select_dtypes(include="number")
            if num_df.empty:
                return "No hay columnas numéricas en el archivo para K-Medias."

            num_df_ajustado = aplicar_ajuste(num_df, tipo_ajuste)

            if tipo_ajuste == "algoritmico":
                X = StandardScaler().fit_transform(num_df_ajustado)
            else:
                X = num_df_ajustado.values

            try:
                modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
                etiquetas = modelo.fit_predict(X)
            except Exception as e:
                return f"Error al ejecutar K-Medias: {e}"

            centros = modelo.cluster_centers_.flatten()
            orden = np.argsort(centros)
            nuevo_cluster = {orden[i]: i for i in range(len(orden))}
            etiquetas_ordenadas = np.array([nuevo_cluster[e] for e in etiquetas])

            df["cluster"] = etiquetas_ordenadas

            table_html = df.head(50).to_html(index=False)

            return render_template(
                "result.html",
                title=f"Resultado K-Medias (K = {k}, ajuste = {tipo_ajuste})",
                table_html=table_html
            )

        elif algoritmo == "kmodas":
            try:
                km = KModes(n_clusters=k, init="Huang", n_init=5, verbose=0)
                etiquetas = km.fit_predict(df.astype(str))
            except Exception as e:
                return f"Error al ejecutar K-Modas: {e}"

            if "Clase" in df.columns:
                centros = km.cluster_centroids_
                idx_clase = df.columns.get_loc("Clase")
                modos = np.array([c[idx_clase] for c in centros])
                orden = np.argsort(modos)
                nuevo_cluster = {orden[i]: i for i in range(len(orden))}
                etiquetas_ordenadas = np.array([nuevo_cluster[e] for e in etiquetas])
            else:
                etiquetas_ordenadas = etiquetas

            df["cluster"] = etiquetas_ordenadas

            table_html = df.head(50).to_html(index=False)

            return render_template(
                "result.html",
                title=f"Resultado K-Modas (K = {k})",
                table_html=table_html
            )

        elif algoritmo == "chiagrup":
            num = df.select_dtypes(include="number").columns
            cat = df.select_dtypes(exclude="number").columns
            if len(num) == 0 or len(cat) == 0:
                return "Para Chi-Agrupamiento necesito al menos una columna numérica y una categórica."

            attr = num[0]
            cls = cat[0]

            if chi_threshold_str:
                try:
                    threshold = float(chi_threshold_str)
                except ValueError:
                    return "Umbral χ² no es un número válido."
            else:
                threshold = None

            try:
                df_res, inter_df, usado, _, _ = chi_merge(df, attr, cls, threshold)
            except Exception as e:
                return f"Error en Chi-Agrupamiento: {e}"

            table_html = (
                "<h2>Resumen de intervalos</h2>"
                + inter_df.to_html(index=False)
                + "<h2>Muestra de datos agrupados</h2>"
                + df_res.head(50).to_html(index=False)
            )

            return render_template(
                "result.html",
                title=f"Resultado Chi-Agrupamiento (umbral χ² = {usado:.3f})",
                table_html=table_html
            )

        elif algoritmo == "arbol":
            etiqueta_objetivo = target_col if target_col else df.columns[-1]
            if etiqueta_objetivo not in df.columns:
                return f"La columna objetivo '{etiqueta_objetivo}' no existe en el CSV."

            X_num = df.select_dtypes(include="number")
            X_no_num = df.select_dtypes(exclude="number")

            if not X_num.empty:
                X_num_aj = aplicar_ajuste(X_num, tipo_ajuste)
                X_all = pd.concat([X_num_aj, X_no_num], axis=1)
            else:
                X_all = df.copy()

            X_all = X_all.drop(columns=[etiqueta_objetivo], errors="ignore")
            X_all = pd.get_dummies(X_all, drop_first=True)
            y = df[etiqueta_objetivo]

            if test_size_str:
                try:
                    test_size = float(test_size_str)
                except ValueError:
                    return "El tamaño del test no es un número válido."
            else:
                test_size = 0.3

            if max_depth_str:
                try:
                    max_depth = int(max_depth_str)
                except ValueError:
                    return "La profundidad máxima debe ser un entero."
            else:
                max_depth = None

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_all, y, test_size=test_size, random_state=42
                )
                arbol = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                arbol.fit(X_train, y_train)
                y_pred = arbol.predict(X_test)
            except Exception as e:
                return f"Error al entrenar el árbol de decisión: {e}"

            acc = accuracy_score(y_test, y_pred)
            df_test = X_test.copy()
            df_test["real"] = y_test
            df_test["pred"] = y_pred

            table_html = (
                f"<p><strong>Exactitud (accuracy):</strong> {acc:.3f}</p>"
                + df_test.head(50).to_html(index=False)
            )

            return render_template(
                "result.html",
                title=f"Árbol de decisiones (ajuste = {tipo_ajuste})",
                table_html=table_html
            )

        elif algoritmo == "soloajuste":
            num_df = df.select_dtypes(include="number")
            no_num_df = df.select_dtypes(exclude="number")

            if num_df.empty:
                return "No hay columnas numéricas para ajustar."

            num_df_aj = aplicar_ajuste(num_df, tipo_ajuste)
            df_ajustado = pd.concat([num_df_aj, no_num_df], axis=1)

            table_html = df_ajustado.head(50).to_html(index=False)

            return render_template(
                "result.html",
                title=f"Datos normalizados (ajuste = {tipo_ajuste})",
                table_html=table_html
            )

        else:
            return f"Algoritmo '{algoritmo}' no reconocido. Revisa el formulario."

    except Exception as e:
        return f"Ocurrió un error raro :): {e}"


if __name__ == "__main__":
    app.run(debug=True)
