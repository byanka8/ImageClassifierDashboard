# Problem Set 4: Image Classifier Dashboard

A Python-based gRPC program developed to simulate client-server distributed system. This project fulfills the requirements of Problem Set 4 in the Distributive Computing course.

## Overview of the Program

- The dashboard acts as a the server displaying live updates of the training models. This is designed using Dear PyGUI python library, dividing the windows into four sections: Batch of images, predictions, ground truths, and a chart for the training loss.
- Meanwhile, the Image Classifier acts as the client, utilizing CNN model and sending live progress to the server.


## Instructions on Running the Program

### 1. Create a virtual environment

```bash
cd gRPC
python -m venv venv
```

### 2. Activate the virtual environment

**On Linux / macOS:**

```bash
source venv/bin/activate
```

**On Windows:**

```bash
venv\Scripts\activate
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the program

```bash
// Launch two cmd or terminals (server and client)
// Make sure to activate venv on both terminals
// Start running the server before the client

// Terminal 1: Running the Server
cd server
python dashboard_server.py

// Terminal 2: Running the Client
cd client
python training_client.py
```
