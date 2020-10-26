import tensorflow as tf
import numpy as np
from tensorflow import keras

# GRADED FUNCTION: house_model
def house_model(y_new):
    # Relation: 5 + 5*number of bedrooms 
    xs = [] # Number of bed rooms
    ys = [] # Price according to the number of bed rooms
    
    # When I put a biger sample size, the model fails to give a result. Numbers are too big or too small ...
    for i in range(10):
        xs.append(i)
        ys.append((5+5*i)/10)
    
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    print(xs)
    print(ys)
    
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=1000)
    return model.predict(y_new)[0]

prediction = house_model([7.0])
print(prediction)