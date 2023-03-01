from keras.datasets import mnist  # MNIST dataset
from keras.models import Model  # specifying and training a neural network
from keras.layers import Input, Dense  # specifying and training a neural network
from keras.utils import np_utils  # images preprocessing
import matplotlib.pyplot as plt  # images input

batch_size = 512  # 512 examples from the train dataset per each iteration
num_epochs = 100  # 100 training passes on the train dataset
hidden_size = 512  # each hidden layer has 512 neurons

num_train = 60000  # MNIST database contains 60,000 training images
num_test = 10000  # MNIST database contains 10,000 testing images

height, width, depth = 28, 28, 1  # MNIST pic size 28x28 grey scale
num_classes = 10  # 10 classes - 1 class for 1 number

(X_train, y_train), (X_test, y_test) = mnist.load_data()  # data downloading from MNIST

X_train = X_train.reshape(num_train, height * width)  # to the one-dimensional data (because we work with Dense
# fully - connected layers, so the inputs must be vectors not tensors so the inputs must be transferred from the pics
# to the chain of numbers)
X_test = X_test.reshape(num_test, height * width)  # to the one-dimensional data
X_train = X_train.astype('float32')  # to the numbers
X_test = X_test.astype('float32')  # to the numbers
X_train /= 255  # Data normalization [0, 255] because we work with the shades of grey
X_test /= 255  # Data normalization [0, 255] because we work with the shades of grey
Y_train = np_utils.to_categorical(y_train, num_classes)  # to the "one hot encoding"
Y_test = np_utils.to_categorical(y_test, num_classes)  # to the "one hot encoding"

# Layers creating

inp = Input(shape=(height * width,))  # Input
hidden_1 = Dense(hidden_size, activation='relu')(inp)  # First hidden layer - fully -connected with ReLU
hidden_2 = Dense(hidden_size, activation='relu')(hidden_1)  # Second hidden layer - fully -connected with ReLU
hidden_3 = Dense(hidden_size, activation='relu')(hidden_2)  # Third hidden layer - fully -connected with ReLU
hidden_4 = Dense(hidden_size, activation='relu')(hidden_3)  # Fourth hidden layer - fully -connected with ReLU
hidden_5 = Dense(hidden_size, activation='relu')(hidden_4)  # Fifth hidden layer - fully -connected with ReLU
hidden_6 = Dense(hidden_size, activation='relu')(hidden_5)  # Six hidden layer - fully -connected with ReLU
hidden_7 = Dense(hidden_size, activation='relu')(hidden_6)  # Seventh hidden layer - fully -connected with ReLU
hidden_8 = Dense(hidden_size, activation='relu')(hidden_7)  # Eight hidden layer - fully -connected with ReLU
hidden_9 = Dense(hidden_size, activation='relu')(hidden_8)  # Ninth hidden layer - fully -connected with ReLU
hidden_10 = Dense(hidden_size, activation='relu')(hidden_9)  # Tens hidden layer - fully -connected with ReLU
out = Dense(num_classes, activation='softmax')(hidden_10)  # Output with softmax

model = Model(inputs=inp, outputs=out)  # It is enough to determine the parameters of the input and output layers
# to determine the model
model.summary()

model.compile(loss='categorical_crossentropy',  # loss function
              optimizer='adam',  # Adam
              metrics=['accuracy'])  # accuracy - to estimate the accuracy (not for the training), equals
# to the numbers of the well-defined classes

model.fit(X_train, Y_train,  # Train the model
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1)
model.evaluate(X_test, Y_test, verbose=1)

# 10 test examples VS images
f, mytest = plt.subplots(10,1)
for i in range(num_test-10, num_test):
    mytest[i-(num_test-10)].imshow((X_test[i].reshape(28,28)))
    print(model.predict(X_test[i].reshape(1,784)))
    plt.imshow(model.predict(X_test[i].reshape(1,784)))
plt.show()
