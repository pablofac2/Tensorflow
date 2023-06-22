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


'''
DOS CAPAS OCULTAS CON 3 NEURONAS CADA UNA, 2 ENTRADAS, 1 SALIDA
'''

humidity = np.array([40, 70, 90, 60, 20, 10, 5], dtype=float)  # New input data

# Define the model architecture
input1 = tf.keras.layers.Input(shape=[1], name='celsius_input')
input2 = tf.keras.layers.Input(shape=[1], name='humidity_input')
concatenated_inputs = tf.keras.layers.Concatenate()([input1, input2])
oculta1 = tf.keras.layers.Dense(units=3)(concatenated_inputs)
oculta2 = tf.keras.layers.Dense(units=3)(oculta1)
salida = tf.keras.layers.Dense(units=1)(oculta2)
modelo = tf.keras.Model(inputs=[input1, input2], outputs=salida)

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = "mean_squared_error"
)

print("Comenzar entrenamiento...")
historial = modelo.fit([celsius, humidity], fahrenheit, epochs=100, verbose=False)
print("Modelo entrenado!")

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show(block=True)

print("Hagamos una predicción!")
resultado = modelo.predict([np.array([100.0]), np.array([0.0])])  # Provide both inputs for prediction
print(f"El resultado es {resultado} fahrenheit!")

print("Valores internos del modelo")
'''
print(oculta1.weights)
print(oculta2.weights)
print(salida.weights)
'''
print(modelo.get_layer("dense_4").get_weights())  # Access oculta1 layer weights
print(modelo.get_layer("dense_5").get_weights())  # Access oculta2 layer weights
print(modelo.get_layer("dense_6").get_weights())  # Access salida layer weights
