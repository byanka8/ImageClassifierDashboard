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
pip install pyinstaller
```
### 4. Generate the .exe files

```bash
# DASHBOARD_SERVER
pyinstaller --onefile dashboard_server.py ^ --add-data "proto;proto"

# TRAINING_CLIENT
pyinstaller --onefile training_client.py ^
    --add-data "proto;proto" ^
    --hidden-import numpy.core._multiarray_umath ^
    --hidden-import numpy.core.multiarray ^
    --hidden-import numpy.linalg.lapack_lite ^
    --hidden-import numpy.random._common ^
    --hidden-import numpy.random._mt19937 ^
    --hidden-import numpy.random._pcg64 ^
    --hidden-import numpy.random._philox ^
    --hidden-import numpy.random._generator
```

### 5. Run the program
Run the program through terminals or simply click the .exe files under the dist folder.
```bash
# Launch two cmd or terminals (server and client)
# Make sure to activate venv on both terminals
# Start running the server before the client

# Terminal 1: Running the Server
cd dist
dashboard_server.exe

# Terminal 2: Running the Client
cd dist
training_client.exe
```
