import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.layers import Dense, Activation,Input
from keras.optimizers import RMSprop
from keras.models import Model

#load data
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('./mnist',one_hot=True)
X_train,y_train=mnist.train.images,mnist.train.labels
X_test,y_test=mnist.test.images,mnist.test.labels
# X_train.shape=(55000,784) y_train.shape=(55000,10)
# X_test.shape=(10000,784)  y_test.shape=(10000,10)

#Defining net structure
inputs=Input(shape=(784,))
dense1=Dense(64,activation='relu')(inputs)
prediction=Dense(10,activation='softmax')(dense1)

model=Model(inputs=inputs,outputs=prediction)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#training
model.fit(X_train,y_train,batch_size=64,epochs=2)

#testing
print("\n Testing--------")
loss ,accuracy=model.evaluate(X_test,y_test)
print('test loss: ', loss)
print('test accuracy: ', accuracy)







