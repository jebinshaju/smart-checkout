# Smart Checkout
**Smart Checkout: Real-Time Automated Billing System Using YOLO**

### Project Description
**Smart Checkout** is a demonstration project that showcases an automated billing system using real-time object detection powered by YOLO and OpenCV. Designed for educational purposes, this system can identify common supermarket items in a webcam feed, calculate their prices, and display an invoice in a user-friendly sidebar. The sidebar includes a total cost calculator, options to save the invoice as a PDF, and a quit button to exit the program.

### Disclaimer
**This is a prototype developed for learning and demonstration purposes only. It is not intended for production use. The item detection and prices are simulated and do not reflect actual values.**

---

## Features
- **Real-Time Object Detection:** Detects predefined items using YOLO model in real-time through a webcam.
- **Dynamic Billing Display:** Lists detected items with their prices and calculates the total cost automatically.
- **PDF Invoice Generation:** Exports a summary of detected items and total cost as a PDF.
- **Interactive User Interface:** Displays detected items in a sidebar with options to save the invoice and exit the program.

---

## Requirements

Create a `requirements.txt` file with the following dependencies:

```plaintext
opencv-python-headless
numpy
ultralytics
python-dotenv
img2pdf
```

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/jebinshaju/smart-checkout.git
   cd smart-checkout
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration:**
   - Create a `.env` file in the root directory with the path to your YOLO model:
     ```plaintext
     YOLO_MODEL_PATH="path/to/your/yolo/model"
     ```

4. **Download the YOLO Model:**
   - Get the pre-trained YOLO model [here](https://github.com/ultralytics/yolov5) and specify the path in the `.env` file.
   - **Additional Resources:** For further customization, check out the [Ultralytics YOLO repository](https://github.com/ultralytics/ultralytics).

5. **Running the Application:**
   - Run the main script:
     ```bash
     python app.py
     ```
   - The program will open the webcam, detect items, and display a billing sidebar.
   - Use **Save Invoice** to save the PDF or **Quit** to exit.

---

## Usage Notes
- **Supported Items:** This project recognizes a limited list of items (e.g., apples, bananas, milk) specified in the `supermarket_items` dictionary.
- **Saving PDFs:** PDF invoices will be saved in the project directory with a timestamp.
- **Exiting:** Use the sidebar **Quit** button or press 'q' to exit.

---

## Resources
- **YOLO Model and Customization**: [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- **Additional Python Libraries**: Refer to [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/), [img2pdf](https://pypi.org/project/img2pdf/).


### Set Up the YOLOv8 Model

You can download the YOLOv8 models from the Ultralytics repository. Make sure to place the model file (`yolov8l.pt`) in the project directory.

| Model Version | Download Link                                                                                          | Model Size | Suitable for         |
|---------------|-------------------------------------------------------------------------------------------------------|------------|-----------------------|
| YOLOv8n       | [Download](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)                 | 6.2 MB     | Raspberry Pi          |
| YOLOv8s       | [Download](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt)                 | 14.5 MB    | Raspberry Pi          |
| YOLOv8m       | [Download](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt)                 | 25.9 MB    | More capable systems  |
| YOLOv8l       | [Download](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt)                 | 46.2 MB    | High performance      |
| YOLOv8x       | [Download](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt)                 | 88.7 MB    | High performance      |

**Note:** For Raspberry Pi, it is recommended to use `yolov8n.pt` or `yolov8s.pt` for faster performance due to limited resources.
---

### Disclaimer
**This project is a demonstration for educational purposes and not intended for real-world use.**
