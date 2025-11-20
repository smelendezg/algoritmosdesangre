from flask import Flask, render_template, request
from servicios.archivo_servicio import leer_csv
from servicios.algoritmo_servicio import ejecutar_algoritmo
from utils.formateador_resultados import formatear_resultado

app = Flask(__name__)

@app.route('/')
def inicio():
    return render_template('inicio.html')

@app.route('/ejecutar', methods=['POST'])
def run():
    try:
        file = request.files.get('file')
        if not file:
            return "No se subió ningún archivo", 400
        
        df = leer_csv(file)
        
        algoritmo = request.form.get('algorithm')
        tipo_ajuste = request.form.get('ajuste', 'ninguno')
        parametros = obtener_parametros(request.form)
        
        resultado = ejecutar_algoritmo(algoritmo, df, tipo_ajuste, parametros)
        contexto = formatear_resultado(algoritmo, resultado, tipo_ajuste)
        return render_template('resultados.html', **contexto)
    except Exception as e:
            return render_template('inicio.html', error=str(e))


def obtener_parametros(form):
    k_str      = form.get('n_clusters')
    chi_str    = form.get('chi_umbral')
    prof_str   = form.get('profundidad_max')
    prueba_str = form.get('prueba')
    criterio = form.get('criterio', 'gini')
    
    parametros = {
        'k': int(k_str) if k_str else None,
        'chi_umbral': float(chi_str) if chi_str else None,
        'col_objetivo': form.get('target_col'),
        'profundidad_max': int(prof_str) if prof_str else None,
        'prueba': float(prueba_str) if prueba_str else None,
        'criterio': criterio,
        'col_grupo': form.get('col_grupo') or None,
        'col_valor': form.get('col_valor') or None,
        'chi_col_atributo': form.get('chi_col_atributo'),
        'chi_col_clase': form.get('chi_col_clase'),
    }

    if parametros['k']:
        parametros['k'] = int(parametros['k'])
    if parametros['chi_umbral']:
        parametros['chi_umbral'] = float(parametros['chi_umbral'])
    if parametros['profundidad_max']:
        parametros['profundidad_max'] = int(parametros['profundidad_max'])
    if parametros['prueba']:
        parametros['prueba'] = float(parametros['prueba'])
    
    return parametros

def validarDatos():
    return 1

if __name__ == '__main__':
    app.run()