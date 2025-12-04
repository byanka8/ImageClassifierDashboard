import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../proto")))

import grpc
import time
import training_pb2_grpc
import training_pb2
from PIL import Image
import io
import random

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# Fault-tolerant helpers
def safe_grpc_call(call_fn, max_retries=5, base_delay=0.2):
    """Execute a gRPC call with retry and exponential backoff.
       Returns None if permanently failed.
    """
    for retry in range(max_retries):
        try:
            return call_fn()
        except grpc.RpcError as e:
            print(f"[WARN] gRPC call failed: {e}. Retry {retry+1}/{max_retries}.")
            time.sleep(base_delay * (2 ** retry))

    print("[ERROR] gRPC unreachable after retries; skipping send.")
    return None


def wait_for_channel_ready(channel, timeout=3):
    """Check if channel is ready. Return True/False."""
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
        return True
    except grpc.FutureTimeoutError:
        return False

# Main training client
if __name__ == "__main__":

    # Connect to dashboard server
    channel = grpc.insecure_channel('localhost:50051')
    stub = training_pb2_grpc.TrainingStreamStub(channel)

    dashboard_available = wait_for_channel_ready(channel)

    if dashboard_available:
        print("[Client] Connected to dashboard UI.")
    else:
        print("[WARN] Dashboard not available. Continuing without sending data.")

    # Dataset loading
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Build Model
    cnn = models.Sequential([
        tf.keras.Input(shape=(32,32,3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    batch_size = 16
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Rate limit send operations
    SEND_INTERVAL = 0 # can adjust this
    last_send_ts = 0

    # Training loop
    for epoch in range(3):
        print(f"Epoch {epoch+1}")

        for batch_idx, (images, labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                predictions = cnn(images, training=True)
                loss = loss_fn(labels, predictions)

            grads = tape.gradient(loss, cnn.trainable_variables)
            optimizer.apply_gradients(zip(grads, cnn.trainable_variables))

            predicted_classes = np.argmax(predictions.numpy(), axis=1)
            true_classes = np.argmax(labels.numpy(), axis=1)

            # Throttle network sends
            now = time.time()
            should_send = dashboard_available and (now - last_send_ts >= SEND_INTERVAL)

            if should_send:
                # Prepare batch message safely
                batch_messages = []
                for i in range(images.shape[0]):
                    try:
                        img = Image.fromarray((images[i].numpy() * 255).astype('uint8'))
                        buf = io.BytesIO()
                        img.save(buf, format='PNG')
                        img_bytes = buf.getvalue()
                    except Exception as e:
                        print("[WARN] Failed to encode image:", e)
                        img_bytes = b""

                    true_label_text = classes[true_classes[i]]
                    predicted_label_text = classes[predicted_classes[i]]

                    batch_messages.append(training_pb2.ImageData(
                        data=img_bytes,
                        ground_truth=true_label_text,
                        prediction=predicted_label_text
                    ))

                # Safe RPC calls
                def send_batch():
                    return stub.SendBatchUpdate(training_pb2.BatchUpdate(images=batch_messages))

                def send_loss():
                    return stub.SendLossUpdate(training_pb2.LossUpdate(
                        loss=float(loss.numpy()),
                        iteration=batch_idx
                    ))

                # If dashboard closed, retry or diable future sends 
                if safe_grpc_call(send_batch) is None:
                    dashboard_available = False

                if dashboard_available:
                    safe_grpc_call(send_loss)

                last_send_ts = time.time()

            print(f"Batch {batch_idx} loss {loss.numpy():.4f}")