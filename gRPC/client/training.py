import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../proto")))

import grpc
import time
import training_pb2_grpc
import training_pb2
# from proto import training_pb2, training_pb2_grpc
from PIL import Image
import io
import random

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# import streamlit as st


if __name__ == "__main__":

    # Connect to dashboard server
    channel = grpc.insecure_channel('localhost:50051')
    stub = training_pb2_grpc.TrainingStreamStub(channel)

    # Load dataset
    (X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()

    # # Use only the first 1000 images for training
    # X_train = X_train[:608]
    # y_train = y_train[:608]

    # # Use a small test set too
    # X_test = X_test[:112]
    # y_test = y_test[:112]

    # Print dataset information
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)

    # Reshape y_train and y_test

    # y_train = y_train.reshape(-1,)
    # y_test = y_test.reshape(-1,)

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    print("y_train shape: ", y_train.shape)
    print("y_test shape: ", y_test.shape)
    
    classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

    # Normalize training data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Build CNN Model
    cnn = models.Sequential([
        tf.keras.Input(shape=(32,32,3)),
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # compile
    # cnn.compile(optimizer='adam',
    #           loss='sparse_categorical_crossentropy',
    #           metrics=['accuracy'])

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    
    # train model
    # cnn.fit(X_train, y_train, epochs=10)

    # batch training
    batch_size = 16
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    for epoch in range(3):  # example: 3 epochs
        print(f"Epoch {epoch+1}")
        
        for batch_idx, (images, labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                predictions = cnn(images, training=True)
                loss = loss_fn(labels, predictions)
            
            # Update weights
            grads = tape.gradient(loss, cnn.trainable_variables)
            optimizer.apply_gradients(zip(grads, cnn.trainable_variables))
            
            # Access per-batch information
            predicted_classes = np.argmax(predictions.numpy(), axis=1)
            true_classes = np.argmax(labels.numpy(), axis=1)
            
            # print(f"Batch {batch_idx}: Loss = {loss.numpy():.4f}")
            # print(f"Predictions: {predicted_classes}")
            # print(f"Ground truth: {true_classes}")

            # Send batch to dashboard
            batch_messages = []
            for i in range(images.shape[0]):
                img = Image.fromarray((images[i].numpy() * 255).astype('uint8'))
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                
                # Convert numeric labels to text
                true_label_text = classes[true_classes[i]]
                predicted_label_text = classes[predicted_classes[i]]
                
                batch_messages.append(training_pb2.ImageData(
                    data=buf.getvalue(),
                    ground_truth=true_label_text,
                    prediction=predicted_label_text
                ))
            stub.SendBatchUpdate(training_pb2.BatchUpdate(images=batch_messages))

            # Send loss
            stub.SendLossUpdate(training_pb2.LossUpdate(
                loss=loss.numpy(),
                iteration=batch_idx
            ))

            print(f"Sent batch {batch_idx} with loss {loss.numpy():.4f}")
