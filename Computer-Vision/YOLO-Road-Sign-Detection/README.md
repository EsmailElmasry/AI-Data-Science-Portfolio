# 🚦 RoadSign AI — Real-Time Road Sign Detection using YOLOv8

A modern, presentation-ready Streamlit web application that performs live road sign
detection using a custom-trained YOLOv8n model. Built as an AI / Computer Vision
graduation project deliverable.

## Detected Classes

| ID | Class Name  | Display Name  |
|----|-------------|---------------|
| 0  | crosswalk   | Crosswalk     |
| 1  | speedlimit  | Speed Limit   |
| 2  | stop        | Stop          |
| 3  | trafficlight| Traffic Light |

## Project Structure

```
roadsign-ai/
│
├── app.py              # Streamlit application (complete, no placeholders)
├── best.pt              # Your trained YOLOv8n weights
├── requirements.txt      # Python dependencies
└── README.md
```

`best.pt` must sit in the **same folder** as `app.py` — the app loads it automatically
from that location.

## Features

- Modern dark dashboard UI with a road/highway-inspired blue-cyan theme
- Sidebar with model info, confidence & IoU threshold sliders
- Drag-and-drop image upload (JPG / JPEG / PNG)
- Real YOLOv8 inference on `best.pt` — no mock or random detections
- Annotated bounding boxes drawn with OpenCV, color-coded per class
- Results dashboard: total detections, highest confidence, detected classes, processing time
- Clean detection table (rank, sign name, confidence)
- "Detection Summary" bar chart of detections per class (Plotly)
- Side-by-side original vs. detection comparison
- Professional empty state before upload
- Graceful error handling (missing model, invalid image, no detections)
- Model loaded once via `st.cache_resource` — fast on every re-run

## How to Run Locally

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Make sure `best.pt` is in the project folder

Confirm the folder looks like:

```
roadsign-ai/
├── app.py
├── best.pt
├── requirements.txt
└── README.md
```

### 4. Launch the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

## Presenting the Project

Suggested demo flow for your instructor:

1. Show the sidebar — explain model info (YOLOv8n, 4 classes) and let the audience see
   the confidence/IoU sliders.
2. Upload a road scene image containing a few different sign types.
3. Point out the Results Dashboard cards (detections, confidence, classes, speed).
4. Walk through the annotated image and the detection table.
5. Show the "Detection Summary" bar chart to summarize class distribution.
6. Adjust the confidence slider live to show how detections change in real time.

## Notes

- Inference runs on CPU by default unless a CUDA-enabled GPU + matching PyTorch build
  is available in your environment.
- If no signs are detected in an image, the app displays a clear
  **"No road signs detected."** message instead of crashing.
