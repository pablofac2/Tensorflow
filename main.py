import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# IMPORTANTE!!!
# write click on code area click on run current file in interactive window

# pip install tensorflow
# pip install numpy
# pip install matplotlib

# también se pueden instalar las librerías en un entorno virtual
#  dentro de la carpeta del proyecto:
#  ps> python -m venv c:\Users\Pablo\source\repos\Tensorflow\venv
# luego ir a Terminal, New Terminal, y ahí correr pip install tensorflow

print("hola mundo")

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

'''
CON SOLO 2 NEURONAS, 1 DE ENTRADA Y UNA DE SALIDA
'''

# capa de salida 1 neurona:
capa = tf.keras.layers.Dense(units=1, input_shape=[1]) #input_shape registra una capa de entrada con 1 neurona
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = "mean_squared_error"
)

print("Comenzar entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado!")

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show(block=True)

print("Hagamos una predicción!")
resultado = modelo.predict([100.0])
print(f"El resultado es {resultado} fahrenheit!")

print("Valores internos del modelo")
print(capa.get_weights())


'''
DOS CAPAS OCULTAS CON 3 NEURONAS CADA UNA
'''
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1]) #input_shape registra una capa de entrada con 1 neurona
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = "mean_squared_error"
)

print("Comenzar entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=100, verbose=False)
print("Modelo entrenado!")

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show(block=True)

print("Hagamos una predicción!")
resultado = modelo.predict([100.0])
print(f"El resultado es {resultado} fahrenheit!")

print("Valores internos del modelo")
print(oculta1.get_weights())
print(oculta2.get_weights())
print(salida.get_weights())
