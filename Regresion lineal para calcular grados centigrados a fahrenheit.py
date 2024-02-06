
# Importar bibliotecas necesarias
import pandas as pd
import seaborn as sb
from sklearn.linear_model import LinearRegression

# Cargar datos desde un archivo CSV
datos = pd.read_csv("celcius.csv")

# Mostrar información sobre los datos cargados
datos.info()

# Mostrar las primeras filas de los datos
datos.head()

# Crear un gráfico de dispersión utilizando Seaborn
sb.scatterplot(x="celcios", y="fahrenheit", data=datos, hue="fahrenheit", palette="coolwarm")

# Definir características (X) y etiquetas (Y)
X = datos["celcios"]
Y = datos["fahrenheit"]

# Mostrar las etiquetas (Y)
Y

# Procesar las características y etiquetas para que tengan la forma adecuada
X_procesada = X.values.reshape(-1, 1)
Y_procesada = Y.values.reshape(-1, 1)

# Crear un modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo con las características y etiquetas procesadas
modelo.fit(X_procesada, Y_procesada)

# Realizar una predicción para un valor específico (por ejemplo, 8)
prediccion = modelo.predict([[8]])

# Mostrar la predicción
print(f"Para 8 grados Celsius, la predicción es {prediccion[0][0]} grados Fahrenheit")

# Ingresar nuevos grados Celsius para realizar una predicción
celcius = 1233
prediccion = modelo.predict([[celcius]])
print(f"{celcius} grados Celsius son aproximadamente {prediccion[0][0]:.2f} grados Fahrenheit")

# Evaluar el rendimiento del modelo (siendo 1.0 muy preciso)
rendimiento_modelo = modelo.score(X_procesada, Y_procesada)
print(f"El rendimiento del modelo es: {rendimiento_modelo:.2%}")
