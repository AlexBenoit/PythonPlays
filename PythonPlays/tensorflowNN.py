
import tensorflow as tf
import numpy as np

LAYER1_NB_NEURONS = 128

def create_model(input_dimension, output_length):
    width, height = input_dimension

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(width, height)),
        tf.keras.layers.Dense(LAYER1_NB_NEURONS, activation=tf.nn.relu),
        tf.keras.layers.Dense(output_length, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print(train_images.shape)
    print(test_images.shape)

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    model = create_model(input_dimension=(28, 28), output_length=len(class_names))
    #model.fit(train_images, train_labels, epochs=5)

    #test_loss, test_acc = model.evaluate(test_images, test_labels)
    #predictions = model.predict(test_images)

    #print("Test accuracy:", test_acc)