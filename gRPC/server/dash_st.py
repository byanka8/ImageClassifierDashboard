import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../proto")))

import threading
import grpc
from concurrent import futures
import streamlit as st
import training_pb2
import training_pb2_grpc
# from generated import trainingmonitor_pb2, trainingmonitor_pb2_grpc
from PIL import Image
import io

import streamlit as st
import numpy as np
import time

st.set_page_config(layout="wide")
st.title("Live ML Training Dashboard")

ROWS = 2
COLS = 8
TOTAL = ROWS * COLS

latest_batch = []   # list<ImageData>
latest_loss = 0.0

class TrainingStreamServicer(training_pb2_grpc.TrainingStreamServicer):
    def SendBatchUpdate(self, request, context):
        global latest_batch
        latest_batch = request.images
        return training_pb2.Ack(status="ok")

    def SendLossUpdate(self, request, context):
        global latest_loss
        latest_loss = request.loss
        return training_pb2.Ack(status="ok")

def serve_grpc():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    training_pb2_grpc.add_TrainingStreamServicer_to_server(
        TrainingStreamServicer(), server
    )
    server.add_insecure_port("localhost:50051")
    server.start()
    print("gRPC server running on port 50051...")
    server.wait_for_termination()


# --- Simulated data ---
# def get_batch_images(session_idx):
#     # Simulate a batch of images
#     return [f"https://picsum.photos/100?random={session_idx}_{i}_{time.time()}" for i in range(BATCH_SIZE)]

# def get_predictions(session_idx):
#     return f"Pred {np.random.randint(0,100)}"

# def get_groundtruths(session_idx):
#     return f"GT {np.random.randint(0,100)}"

# --- Helper to create section placeholders ---
def create_section(rows, cols):
    section_cells = []
    for _ in range(rows):
        cols_row = st.columns(cols)
        section_cells.extend([col.empty() for col in cols_row])
    return section_cells

# --- Placeholders for sections ---
st.subheader("Batch of images")
input_cells = create_section(ROWS, COLS)

st.subheader("Predictions")
pred_cells = create_section(ROWS, COLS)

st.subheader("Ground Truths")
gt_cells = create_section(ROWS, COLS)

st.subheader("Loss Chart")
loss_history = [0.0]
chart = st.line_chart(loss_history, color=["#483d8b"])

# --- FPS placeholder ---
fps_placeholder = st.empty()
last_time = time.time()

# --- Main loop ---
def main():

    # --- Start the gRPC server ---
    threading.Thread(target=serve_grpc, daemon=True).start()

    global last_time, last_rows

    # Main UI Loop (keeps refreshing)
    while True:

        # --- If real batch is received from gRPC ---
        if latest_batch:
            for i in range(min(len(latest_batch), TOTAL)):
                img = latest_batch[i]

                # Show the image (byte data)
                input_cells[i].image(img.data, width=80)

                # Update prediction text
                pred_cells[i].text(img.prediction)

                # Update ground truth text
                gt_cells[i].text(img.ground_truth)

        # --- Update loss chart ---
        loss_history.append(latest_loss)
        chart.add_rows([latest_loss])

        # --- Update FPS ---
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        fps = 1 / dt if dt > 0 else 0
        fps_placeholder.text(f"FPS: {fps:.2f}")

        # Streamlit requires rerun
        time.sleep(0.05)
        # st.experimental_rerun()

if __name__ == "__main__":
    main()