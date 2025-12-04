import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "proto")))

import io
import time
import threading
import logging
from concurrent import futures as cfutures

import numpy as np
from PIL import Image

import dearpygui.dearpygui as dpg

import grpc
from proto import training_pb2
from proto import training_pb2_grpc

# CONFIG
ROWS = 2
COLS = 8
TOTAL = ROWS * COLS
FPS = 60
IMG_SIZE = 96  # texture size used by the dashboard

UPDATE_LOSS_EVERY = 16
last_loss_iteration = -1

last_image = [None] * TOTAL  # initialize a list to track last displayed image per slot

# shared state updated by gRPC handlers 
latest_batch = None   # will hold list of ImageData messages
latest_loss = 0.0
loss_history = []
step_history = []

# thread-safe lock for state
state_lock = threading.Lock()

# batch update throttling 
last_batch_update = 0.0  # global time of last visual update
BATCH_UPDATE_INTERVAL = 2.0  # seconds between visual batch updates

# gRPC Servicer
class TrainingStreamServicer(training_pb2_grpc.TrainingStreamServicer):
    def SendBatchUpdate(self, request, context):
        """Unary RPC: client sends BatchUpdate(images=[ImageData,...])"""
        global latest_batch
        try:
            with state_lock:
                # store the message list
                latest_batch = list(request.images)
            return training_pb2.Ack(status="ok")
        except Exception as e:
            logging.exception("Error in SendBatchUpdate")
            return training_pb2.Ack(status=f"error: {e}")

    def SendLossUpdate(self, request, context):
        global latest_loss, last_loss_iteration
        if request.iteration % UPDATE_LOSS_EVERY == 0:
            with state_lock:
                latest_loss = float(request.loss)
            last_loss_iteration = request.iteration
        return training_pb2.Ack(status="ok")

def serve_grpc(port=50051):
    server = grpc.server(cfutures.ThreadPoolExecutor(max_workers=4))
    training_pb2_grpc.add_TrainingStreamServicer_to_server(TrainingStreamServicer(), server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    server.start()
    logging.info(f"[gRPC] Server running on port {port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)

# DearPyGUI UI setup
dpg.create_context()

# create a texture registry and dynamic textures for TOTAL tiles
texture_ids = []
with dpg.texture_registry():
    for i in range(TOTAL):
        # initialize with a small blank RGBA image (values in 0..1 flattened)
        blank = np.zeros((IMG_SIZE, IMG_SIZE, 4), dtype=np.float32)
        tid = dpg.add_dynamic_texture(width=IMG_SIZE, height=IMG_SIZE, default_value=blank.flatten().tolist())
        texture_ids.append(tid)

# store widget ids for updates
image_widgets = []
pred_widgets = []
gt_widgets = []

# build viewport and layout
dpg.create_viewport(title="Image Classifier Dashboard", width=1400, height=670)

with dpg.window(label="Image Classifier Dashboard", width=1400, height=670):
    dpg.add_text("Image Classifier Dashboard", color=(200, 200, 255))
    dpg.add_separator()

    # Create a scrollable region
    with dpg.child_window(width=-1, height=-1, autosize_x=True, autosize_y=False):
        
        # FPS display
        dpg.add_text("FPS: 0", tag="fps_text")
        dpg.add_spacer(height=6)

        # Batch images grid
        dpg.add_text("Batch of Images")
        idx = 0
        for r in range(ROWS):
            with dpg.group(horizontal=True):
                for c in range(COLS):
                    # create image widget bound to texture
                    img_widget = dpg.add_image(texture_ids[idx])
                    image_widgets.append(img_widget)
                    idx += 1
        dpg.add_spacer(height=6)

        # Predictions
        dpg.add_text("Predictions")
        for r in range(ROWS):
            with dpg.group(horizontal=True):
                for c in range(COLS):
                    txt = dpg.add_text("", color=(0, 200, 200))
                    pred_widgets.append(txt)
        dpg.add_spacer(height=3)

        # Ground Truths
        dpg.add_text("Ground Truths")
        for r in range(ROWS):
            with dpg.group(horizontal=True):
                for c in range(COLS):
                    txt = dpg.add_text("", color=(200, 200, 0))
                    gt_widgets.append(txt)
        dpg.add_spacer(height=3)

        # Loss chart
        dpg.add_text("Loss Chart")
        with dpg.plot(label="Loss", height=200, width=1200):
            dpg.add_plot_axis(dpg.mvXAxis, label="iteration", auto_fit=True)
            with dpg.plot_axis(dpg.mvYAxis, label="loss", tag="y_axis_tag"):
                loss_series = dpg.add_line_series([], [], label="loss")

        dpg.set_axis_limits("y_axis_tag", -1, 3)

# finish UI setup
dpg.setup_dearpygui()
dpg.show_viewport()

# helper functions
def pil_to_rgba_flat(pil_img, target_size=IMG_SIZE):
    """Convert PIL.Image -> flattened list of floats (RGBA 0..1) for dpg dynamic texture."""
    # Resize to target_size
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    # Resize - preserve aspect ratio: fit into square and paste onto transparent background
    w, h = pil_img.size
    if (w, h) != (target_size, target_size):
        # scale preserving aspect ratio
        pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0  # shape (H,W,4)
    return arr.flatten().tolist()

# FPS trackers
last_time = time.perf_counter()
frame_count = 0
fps_timer = time.perf_counter()

def update_once():
    global frame_count, last_time, fps_timer, loss_history, step_history, last_batch_update

    now = time.perf_counter()
    dt = now - last_time
    last_time = now
    frame_count += 1

    # read shared state
    with state_lock:
        batch_copy = list(latest_batch) if latest_batch else None

    # update images, predictions, GT only every BATCH_UPDATE_INTERVAL seconds
    if batch_copy and (now - last_batch_update >= BATCH_UPDATE_INTERVAL):
        for i, img_data in enumerate(batch_copy[:TOTAL]):
            if img_data != last_image[i]:
                # flat = pil_to_rgba_flat(Image.open(io.BytesIO(img_data.data)), IMG_SIZE)
                try: # If the client sends invalid bytes, PIL will raise
                    img = Image.open(io.BytesIO(img_data.data))
                    flat = pil_to_rgba_flat(img, IMG_SIZE)
                except Exception:
                    logging.exception("Failed to decode image; using blank.")
                    flat = [0.0] * (IMG_SIZE * IMG_SIZE * 4)
                dpg.set_value(texture_ids[i], flat)
                dpg.set_value(pred_widgets[i], f"{i+1:<2}: {img_data.prediction:<10}")
                dpg.set_value(gt_widgets[i], f"{i+1:<2}: {img_data.ground_truth:<10}")
                last_image[i] = img_data
        last_batch_update = now  # reset timer

    # update loss chart every frame
    with state_lock:  # read latest_loss safely
        current_loss = latest_loss
        current_step = last_loss_iteration
    if not step_history or current_step > step_history[-1]: # Only append if new iteration
    # if current_step % 2 == 0:  # append every 2nd received loss
        loss_history.append(current_loss)
        step_history.append(current_step)
        # Update plot
        dpg.set_value(loss_series, [step_history, loss_history])

    # update FPS once per second
    if now - fps_timer >= 1.0:
        fps = frame_count
        frame_count = 0
        fps_timer = now
        dpg.set_value("fps_text", f"FPS: {fps}")

    # Built-in
    # current_fps = dpg.get_frame_rate()
    # dpg.set_value("fps_text", f"FPS: {current_fps:.1f}")

# start gRPC server in background thread
grpc_thread = threading.Thread(target=serve_grpc, kwargs={"port": 50051}, daemon=True)
grpc_thread.start()

# DearPyGUI render loop
try:
    while dpg.is_dearpygui_running():
        try: # if an exception occurs, the program will terminate
            update_once()
        except Exception:
            logging.exception("UI update failed; continuing.")
        dpg.render_dearpygui_frame()
finally:
    dpg.destroy_context()