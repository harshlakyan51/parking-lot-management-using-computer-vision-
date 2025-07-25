
# 🅿️ Smart Parking Spot Detection using Computer Vision

> **Real-time parking space occupancy detection using OpenCV, Scikit-learn, and classical Machine Learning.**
> This project detects whether parking spots are empty or occupied in a video stream using a trained ML model and computer vision techniques.

---

## 🚀 Project Overview

**Problem:**
Manual monitoring of parking spaces is inefficient and often inaccurate. A camera-based automatic parking detection system can solve this with minimal infrastructure.

**Solution:**
We process video input using:

* A static mask for parking area regions.
* Frame differencing to detect changed spots.
* A trained machine learning model to classify each parking spot as **empty** or **occupied**.

---


## 📷 Screenshots 

<img width="1597" height="900" alt="image" src="https://github.com/user-attachments/assets/7bd22f72-371b-4d9f-88f2-1449be3215ef" />



## 🎯 Features

* ✅ Real-time spot detection using pre-recorded video
* ✅ ML model to classify empty vs. filled spots
* ✅ Mask-based region of interest (ROI) processing
* ✅ Intelligent frame differencing to reduce redundant computation
* ✅ Automatic bounding box merging for flexible parking layouts
* ✅ Clean and scalable codebase

---

## 🧠 Technologies Used

| Category             | Tech Stack                  |
| -------------------- | --------------------------- |
| Programming Language | Python 3.x                  |
| CV Library           | OpenCV                      |
| ML                   | Scikit-learn                |
| Image Processing     | Skimage (resize)            |
| Model Storage        | Pickle                      |
| Display              | OpenCV GUI (cv2.imshow)     |
| Video Processing     | Mask + Connected Components |

---

## 📁 Folder Structure

```
├── main.py                   # Main application logic
├── util.py                   # Utility functions (model loading, classification, box merging)
├── model.p                   # classifier model 
├── mask_1920_1080.png        # Binary mask to isolate parking spots
├── samples/
│   └── parking_1920_1080_loop.mp4   # Sample loop video input
```

---

## 🧪 Model Description

The classification model (`model.p`) is trained to distinguish between **occupied** and **empty** parking spots using 15x15 RGB resized patches.

### Input:

* Cropped image of a parking spot (from video)
* Resized to `(15, 15, 3)`, then flattened

### Output:

* `0` for **Empty**
* `1` for **Occupied**

The prediction is used to color each bounding box:

* 🟩 **Green** — Available
* 🟥 **Red** — Occupied

---

### 3. Run the Application

```bash
python main.py
```

To quit the application, press `q` in the video window.

---

## 🛠 How it Works – Step by Step

1. **Mask Preprocessing:**
   Load a binary mask image that marks all valid parking regions.

2. **Connected Components:**
   Use `cv2.connectedComponentsWithStats()` to identify individual parking spot areas.

3. **Box Merging:**
   Nearby or vertically-aligned boxes are merged for flexibility.

4. **Frame Differencing:**
   Every N frames (default 30), calculate per-spot differences with the previous frame to reduce computation.

5. **Classification:**
   For each changed spot, crop the image, resize, flatten, and classify using the ML model.

6. **Visualization:**
   Display results in real time with colored rectangles and total available count.

---

## 🧪 Sample Output

```
Available spots: 12 / 17
```

**Green boxes** indicate available spots, and **red boxes** show occupied ones.

---

## 📈 Performance Optimizations

* Frame sampling to reduce redundant computation.
* Per-ROI difference check before running classification.
* Model size and input resolution optimized for speed.

---

## 📌 Use Cases

* Malls / commercial parking management
* Smart city solutions
* Surveillance-integrated vehicle tracking
* Low-cost retrofitting in existing CCTV setups

---

## 🧠 Future Work

* Integrate deep learning (e.g., CNN, YOLOv8) for improved accuracy
* Real-time object detection without fixed masks
* Raspberry Pi or edge device deployment
* Parking analytics dashboard with plots

---




