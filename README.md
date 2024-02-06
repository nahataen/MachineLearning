# MachineLearning
Ejercicio de machine learning
# Predicción de Temperatura con Regresión Lineal

Este repositorio contiene un simple script en Python para predecir la temperatura en grados Fahrenheit a partir de la temperatura en grados Celsius utilizando un modelo de regresión lineal. El script utiliza las bibliotecas `pandas`, `seaborn`, y `scikit-learn`.

## Instrucciones de Uso

1. Clona este repositorio:

    ```bash
    git clone https://github.com/tu_usuario/tu-repositorio.git
    ```

2. Asegúrate de tener Python instalado en tu sistema.

3. Instala las dependencias:

    ```bash
    pip install pandas seaborn scikit-learn
    ```

4. Ejecuta el script:

    ```bash
    python temperatura_prediction.py
    ```

## Descripción del Código

El script realiza las siguientes operaciones:

1. Importa las bibliotecas necesarias:

    ```python
    import pandas as pd
    import seaborn as sb
    from sklearn.linear_model import LinearRegression
    ```

2. Carga datos desde un archivo CSV y muestra información sobre los datos:

    ```python
    datos = pd.read_csv("celcius.csv")
    datos.info()
    ```

3. Muestra las primeras filas de los datos:

    ```python
    datos.head()
    ```

4. Crea un gráfico de dispersión utilizando Seaborn:

    ```python
    sb.scatterplot(x="celcios", y="fahrenheit", data=datos, hue="fahrenheit", palette="coolwarm")
    ```

5. Define características (X) y etiquetas (Y):

    ```python
    X = datos["celcios"]
    Y = datos["fahrenheit"]
    ```

6. Procesa las características y etiquetas para que tengan la forma adecuada:

    ```python
    X_procesada = X.values.reshape(-1, 1)
    Y_procesada = Y.values.reshape(-1, 1)
    ```

7. Crea un modelo de regresión lineal:

    ```python
    modelo = LinearRegression()
    ```

8. Entrena el modelo con las características y etiquetas procesadas:

    ```python
    modelo.fit(X_procesada, Y_procesada)
    ```

9. Realiza una predicción para un valor específico (por ejemplo, 8):

    ```python
    prediccion = modelo.predict([[8]])
    print(f"Para 8 grados Celsius, la predicción es {prediccion[0][0]} grados Fahrenheit")
    ```

10. Ingresa nuevos grados Celsius para realizar una predicción y muestra el resultado:

    ```python
    celcius = 1233
    prediccion = modelo.predict([[celcius]])
    print(f"{celcius} grados Celsius son aproximadamente {prediccion[0][0]:.2f} grados Fahrenheit")
    ```

11. Evalúa el rendimiento del modelo:

    ```python
    rendimiento_modelo = modelo.score(X_procesada, Y_procesada)
    print(f"El rendimiento del modelo es: {rendimiento_modelo:.2%}")
    ```

[Creado por Nahataen](https://github.com/nahataen)
