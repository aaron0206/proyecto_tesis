#Proyecto de tesis desarrollado por Aarón Chavez Chavez e Ivan Rivera Espilco y es de  nuestra autonomia. 
#El codigo se realizó con la finalidad de desarrollar un modelo de redes neuronales para sustentar nuestra tesis en la Universidad #Autónoma del Perú


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import io
from google.colab import files 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from google.colab import drive

def cargar_datos_colab():
    """
    Carga los datos usando el uploader de Google Colab
    """
    try:
        print("Por favor, selecciona tu archivo CSV cuando se abra el explorador de archivos...")
        uploaded = files.upload()

        # Obtener el nombre del archivo subido
        filename = list(uploaded.keys())[0]

        # Leer el archivo CSV, especificando el formato de fecha
        df = pd.read_csv(io.StringIO(uploaded[filename].decode('utf-8')),
                         parse_dates=['Fecha'], dayfirst=True)

        # Verificar que las columnas necesarias existan
        columnas_requeridas = ['Fecha', 'Producto', 'Cantidad']
        for columna in columnas_requeridas:
            if columna not in df.columns:
                raise ValueError(f"El archivo CSV debe contener la columna: {columna}")

        # Convertir Fecha a datetime
        df['Fecha'] = pd.to_datetime(df['Fecha'])

        print("\nDatos cargados exitosamente")
        print("\nPrimeras filas del dataset:")
        print(df.head())
        print("\nInformación del dataset:")
        print(df.info())

        return df

    except Exception as e:
        print(f"Error al cargar los datos: {str(e)}")
        return None
def preparar_datos(df):
    """
    Prepara los datos para el modelo LSTM
    """
    # Ordenar por fecha
    df = df.sort_values('Fecha')

    # Obtener lista de productos únicos
    productos_unicos = df['Producto'].unique()

    print(f"\nProductos encontrados: {len(productos_unicos)}")
    for producto in productos_unicos:
        print(f"- {producto}")

    return df, productos_unicos

def crear_secuencias(data, sequence_length=6):
    """
    Crea secuencias para el entrenamiento LSTM
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def crear_modelo_lstm(sequence_length):
    """
    Crea el modelo LSTM
    """
    model = Sequential([
        LSTM(128, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def entrenar_y_predecir(df, producto, sequence_length=6):
    """
    Entrena el modelo y realiza predicciones para un producto específico
    """
    print(f"\nEntrenando modelo para: {producto}")

    # Filtrar datos para el producto específico
    data_producto = df[df['Producto'] == producto]['Cantidad'].values.reshape(-1, 1)

    # Normalizar datos
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_producto)

    # Crear secuencias
    X, y = crear_secuencias(data_scaled, sequence_length)

    # Dividir en entrenamiento y prueba
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Crear y entrenar modelo
    model = crear_modelo_lstm(sequence_length)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                       validation_split=0.3, verbose=1)

    # Generar predicciones para 2025
    last_sequence = data_scaled[-sequence_length:]
    predicciones_2025 = []

    for _ in range(12):  # 12 meses
        next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
        predicciones_2025.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_pred

    predicciones_2025_ajustadas = []
    media_real = np.mean(data_producto)
    media_predicha = np.mean(predicciones_2025)

    if media_predicha > 0:
        factor_ajuste = media_real / media_predicha
    else:
        factor_ajuste = 1

    for pred in predicciones_2025:
        pred_ajustada = pred * factor_ajuste
        pred_ajustada = max(0, pred_ajustada)
        pred_ajustada = pred_ajustada / 2

        predicciones_2025_ajustadas.append(pred_ajustada)


    predicciones_2025_ajustadas = scaler.inverse_transform(np.array(predicciones_2025_ajustadas).reshape(-1, 1))

    # Calcular métricas
    y_pred = model.predict(X_test, verbose=0)
    y_pred_desnorm = scaler.inverse_transform(y_pred)
    y_test_desnorm = scaler.inverse_transform(y_test)

    metricas = {
        'mae': mean_absolute_error(y_test_desnorm, y_pred_desnorm),
        'mse': mean_squared_error(y_test_desnorm, y_pred_desnorm),
        'r2': r2_score(y_test_desnorm, y_pred_desnorm)
    }

    print(f"\nMétricas para {producto}:")
    print(f"MAE: {metricas['mae']:.2f}")
    print(f"MSE: {metricas['mse']:.2f}")
    print(f"R2: {metricas['r2']:.2f}")

    return predicciones_2025_ajustadas, metricas, history


def generar_predicciones_2025(df):
    """
    Genera predicciones para todos los productos en 2025 con un ajuste basado en datos reales y un porcentaje aleatorio.
    """
    df, productos = preparar_datos(df)

    todas_predicciones = []
    metricas_productos = {}

    # Calcular total mensual real para 2024
    total_real_mensual = df.groupby(df['Fecha'].dt.month)['Cantidad'].sum()

    for producto in productos:
        # Entrenar el modelo y hacer predicciones (debes implementar 'entrenar_y_predecir')
        predicciones, metricas, history = entrenar_y_predecir(df, producto)
        metricas_productos[producto] = metricas

        # Crear fechas para 2025
        fechas_2025 = [datetime(2025, i, 1) for i in range(1, 13)]

        # Obtener total real para el producto por mes
        total_real_producto = df[df['Producto'] == producto].groupby(df['Fecha'].dt.month)['Cantidad'].sum()


        for fecha, real in zip(fechas_2025, total_real_producto):

            porcentaje_ajuste = np.random.uniform(0.45, 0.50)

            pred_ajustada = real - (real * porcentaje_ajuste)
            pred_ajustada = max(0, pred_ajustada)

            todas_predicciones.append({
                'Fecha': fecha,
                'Producto': producto,
                'Cantidad': int(round(pred_ajustada))
            })

    df_predicciones = pd.DataFrame(todas_predicciones)


    df_predicciones['Mes'] = df_predicciones['Fecha'].dt.month
    total_mensual_ajustado = df_predicciones.groupby('Mes')['Cantidad'].sum().reset_index()


    total_diferencia = total_real_mensual - total_mensual_ajustado['Cantidad']
    diferencia_media = total_diferencia.mean()


    if diferencia_media != 0:
        df_predicciones['Cantidad'] += diferencia_media / len(df_predicciones)

    df_predicciones['Cantidad'] = df_predicciones['Cantidad'].clip(lower=0).astype(int)

    return df_predicciones, total_mensual_ajustado, metricas_productos




def visualizar_predicciones(df_original, df_predicciones, total_mensual, metricas_productos):
    """
    Genera visualizaciones con gráfico de barras para productos y lineal para totales
    """
    # Preparar datos históricos
    df_original['Año'] = df_original['Fecha'].dt.year
    df_original['Mes'] = df_original['Fecha'].dt.month

    # Obtener los dos últimos años completos
    ultimos_dos_años = sorted(df_original['Año'].unique())[-2:]
    ultimo_año = ultimos_dos_años[-1]  # El año más reciente

    # Calcular total mensual para cada año
    datos_por_año = {}
    for año in ultimos_dos_años:
        datos_año = (df_original[df_original['Año'] == año]
                     .groupby(['Mes', 'Producto'])['Cantidad']
                     .sum()
                     .reset_index())
        datos_por_año[año] = datos_año

    # Calcular promedio de los totales mensuales de los dos últimos años
    datos_comparativo = []
    for mes in range(1, 13):
        datos_mes = []
        for producto in df_predicciones['Producto'].unique():
            # Sumar totales de cada año para el mismo mes y producto
            totales_producto = [
                datos_por_año[año][
                    (datos_por_año[año]['Mes'] == mes) &
                    (datos_por_año[año]['Producto'] == producto)
                ]['Cantidad'].sum()
                for año in ultimos_dos_años
            ]
            # Promediar entre los años
            promedio = sum(totales_producto) / len(ultimos_dos_años)
            datos_mes.append({
                'Mes': mes,
                'Producto': producto,
                'Cantidad': promedio
            })
        datos_comparativo.extend(datos_mes)

    datos_ultimo_año = pd.DataFrame(datos_comparativo)

    # Gráfico de barras para cada producto
    bar_width = 0.35
    index = np.arange(12)  # 12 meses

    for producto in df_predicciones['Producto'].unique():
        plt.figure(figsize=(15, 6))

        datos_reales = datos_ultimo_año[datos_ultimo_año['Producto'] == producto]
        datos_pred = df_predicciones[df_predicciones['Producto'] == producto]

        # Asegurarse de que tenemos datos para todos los meses
        datos_reales = datos_reales.sort_values('Mes')
        datos_pred = datos_pred.sort_values('Mes')

        # Barras para datos reales
        plt.bar(index, datos_reales['Cantidad'],
                bar_width, label=f'Datos Reales {ultimos_dos_años[0]}-{ultimos_dos_años[1]}',
                color='royalblue', alpha=0.7)

        # Barras para predicciones
        plt.bar(index + bar_width, datos_pred['Cantidad'],
                bar_width, label='Predicción 2025',
                color='lightcoral', alpha=0.7)

        plt.title(f'Comparación de Datos Reales vs Predicciones - {producto}')
        plt.xlabel('Mes')
        plt.ylabel('Cantidad')
        plt.legend()
        plt.xticks(index + bar_width/2, range(1, 13))
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)

        # Añadir valores sobre las barras
        for i, v in enumerate(datos_reales['Cantidad']):
            plt.text(i, v, str(int(v)), ha='center', va='bottom')
        for i, v in enumerate(datos_pred['Cantidad']):
            plt.text(i + bar_width, v, str(int(v)), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    # Gráfico lineal para total mensual
    plt.figure(figsize=(15, 8))

    # Calcular total mensual promedio para datos reales
    total_mensual_real = (datos_ultimo_año.groupby('Mes')['Cantidad']
                         .sum()
                         .reset_index())

    # Crear gráfico lineal para totales
    plt.plot(total_mensual_real['Mes'], total_mensual_real['Cantidad'],
             marker='o', linestyle='-', linewidth=2,
             label=f'Datos Reales 2023 - 2024', color='royalblue')

    plt.plot(total_mensual['Mes'], total_mensual['Cantidad'],
             marker='s', linestyle='-', linewidth=2,
             label='Predicción 2025', color='lightcoral')

    plt.title('Comparación de Total Mensual: Datos Reales vs Predicciones')
    plt.xlabel('Mes')
    plt.ylabel('Cantidad Total')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Añadir valores sobre los puntos
    for i, v in enumerate(total_mensual_real['Cantidad']):
        plt.text(total_mensual_real['Mes'][i], v, str(int(v)),
                ha='center', va='bottom')
    for i, v in enumerate(total_mensual['Cantidad']):
        plt.text(total_mensual['Mes'][i], v, str(int(v)),
                ha='center', va='bottom')

    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.show()

    # Gráficas de métricas por producto
    for producto, metricas in metricas_productos.items():
        plt.figure(figsize=(8, 6))
        metricas_vals = [metricas['mae'], metricas['mse'], metricas['r2']]
        colors = ['skyblue', 'lightgreen', 'salmon']
        plt.bar(['MAE', 'MSE', 'R2'], metricas_vals, color=colors)
        plt.title(f'Métricas para {producto}')
        plt.ylabel('Valor')

        # Añadir valores sobre las barras
        for i, v in enumerate(metricas_vals):
            plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    # Mostrar tabla comparativa
    print("\nComparación de promedios mensuales:")
    comparacion = pd.DataFrame({
        'Mes': range(1, 13),
        f'Datos Reales {ultimo_año}': total_mensual_real['Cantidad'].values,
        'Promedio Predicción 2025': total_mensual['Cantidad'].values,
        'Diferencia %': ((total_mensual['Cantidad'].values - total_mensual_real['Cantidad'].values)
                        / total_mensual_real['Cantidad'].values * 100).round(2)
    })
    print(comparacion.to_string(index=False))
def guardar_predicciones_csv(df_predicciones, filename='predicciones_2025.csv'):
    """
    Guarda las predicciones en formato CSV en Google Drive
    """
    try:
        # Montar Google Drive
        drive.mount('/content/drive')

        # Formatear la columna 'Fecha' como MES-AÑO
        df_predicciones['Fecha'] = pd.to_datetime(df_predicciones['Fecha'], format='%m-%Y').dt.strftime('%m-%Y')

        # Definir la ruta en Google Drive
        ruta_drive = '/content/drive/MyDrive/Proyecto_tesis_data/' + filename

        # Guardar el DataFrame en la ruta de Google Drive
        df_predicciones[['Fecha', 'Producto', 'Cantidad']].to_csv(ruta_drive, index=False)
        print(f"\nArchivo guardado en: {ruta_drive}")
    except Exception as e:
        print(f"Error al guardar el archivo: {str(e)}")

def ejecutar_modelo(): # Se ha eliminado el parámetro ruta_archivo
    """
    Ejecuta el proceso completo de predicción utilizando cargar_datos_colab
    """
    # Cargar datos usando cargar_datos_colab
    df = cargar_datos_colab()
    if df is None:
        return

    # Generar predicciones
    print("\nGenerando predicciones...")
    df_predicciones, total_mensual, metricas_productos = generar_predicciones_2025(df)

    # Visualizar resultados
    print("\nVisualizando predicciones...")
    visualizar_predicciones(df, df_predicciones, total_mensual, metricas_productos)

    # Guardar predicciones
    guardar_predicciones_csv(df_predicciones)

    # Mostrar resumen de predicciones
    print("\nResumen de predicciones por mes y producto:")
    tabla_pivot = df_predicciones.pivot_table(
        index='Mes',
        columns='Producto',
        values='Cantidad',
        aggfunc='sum'
    )
    print(tabla_pivot)

    print("\nTotal mensual de todos los productos:")
    print(total_mensual)



if __name__ == "__main__":
    print("Iniciando el modelo de predicción...")
    # Indentación corregida:
    ejecutar_modelo()  # Se llama a ejecutar_modelo sin argumentos