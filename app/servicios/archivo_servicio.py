import pandas as pd

def leer_csv(archivo):
    if not archivo or archivo.filename == "":
        raise ValueError("No subiste archivo CSV.")
    try:
        df = pd.read_csv(archivo, na_values=['?'])
    except Exception as e:
        raise ValueError(f"Error leyendo el CSV (¿es realmente un .csv?): {e}")
    return df    
        
        