import cv2
import numpy as np
from ultralytics import YOLO
import os
from dotenv import load_dotenv
import datetime
import tempfile
import img2pdf

# Load environment variables from .env file
load_dotenv()

# Load the YOLO model (path from .env)
model = YOLO(os.getenv('YOLO_MODEL_PATH'))

cap = cv2.VideoCapture(0)

detected_items = set()

# List of supermarket items with their prices in Indian Rupees
supermarket_items = {
    'apple': 50.00,
    'banana': 30.00,
    'orange': 60.00,
    'milk': 60.00,
    'bread': 40.00,
    'cheese': 250.00,
    'eggs': 180.00,
    'chocolate': 150.00,
    'tomato': 40.00,
    'potato': 20.00
}

save_button_coords = None
quit_button_coords = None

exit_program = False

# Function to save the invoice as PDF
def save_invoice_as_pdf(sidebar_image):
    temp_image_path = tempfile.mktemp(suffix='.png')
    cv2.imwrite(temp_image_path, sidebar_image)
    
    # Convert the image to PDF
    pdf_path = f"Invoice_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(temp_image_path))
    os.remove(temp_image_path)
    print(f"Invoice saved as {pdf_path}")

def mouse_event(event, x, y, flags, param):
    global save_button_coords, quit_button_coords, exit_program, sidebar
    frame_width = param['frame_width']

    if event == cv2.EVENT_LBUTTONDOWN:
        x_sidebar = x - frame_width

        if x_sidebar >= 0:
            if save_button_coords[0][0] <= x_sidebar <= save_button_coords[1][0] and save_button_coords[0][1] <= y <= save_button_coords[1][1]:
                save_invoice_as_pdf(sidebar)
            elif quit_button_coords[0][0] <= x_sidebar <= quit_button_coords[1][0] and quit_button_coords[0][1] <= y <= quit_button_coords[1][1]:
                exit_program = True

cv2.namedWindow('Automatic Billing System')

ret, frame = cap.read()
if not ret:
    print("Failed to read from webcam.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

frame_width = frame.shape[1]

cv2.setMouseCallback('Automatic Billing System', mouse_event, param={'frame_width': frame_width})

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model on the frame
    results = model(frame)

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = box.conf[0]

            # Check if the detected object is in our supermarket items list
            if confidence > 0.5 and label in supermarket_items:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 205, 50), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}',
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (50, 205, 50), 2)

                # Add the item to detected_items if it's not already there
                if label not in detected_items:
                    detected_items.add(label)

    # Create a sidebar for the invoice
    sidebar_width = 300
    sidebar_height = frame.shape[0]
    sidebar = np.ones((sidebar_height, sidebar_width, 3), dtype=np.uint8) * 255

    # Draw a border around the sidebar
    cv2.rectangle(sidebar, (0, 0), (sidebar_width - 1, sidebar_height - 1), (0, 0, 0), 2)

    # Display the items and total cost on the sidebar
    y_offset = 40
    total_cost = 0

    # Invoice Header
    cv2.putText(sidebar, 'INVOICE', (80, y_offset),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 128), 2)
    y_offset += 40
    cv2.line(sidebar, (10, y_offset), (sidebar_width - 10, y_offset), (0, 0, 0), 2)
    y_offset += 30

    # Display each unique item once
    for item in sorted(detected_items):
        price = supermarket_items[item]
        text = f'{item.capitalize()}: Rs.{price:.2f}'
        cv2.putText(sidebar, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y_offset += 30
        total_cost += price

    y_offset += 10
    cv2.line(sidebar, (10, y_offset), (sidebar_width - 10, y_offset), (0, 0, 0), 2)
    y_offset += 40

    # Display total cost
    cv2.putText(sidebar, f'Total: Rs.{total_cost:.2f}', (10, y_offset),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    y_offset += 60

    # Draw the Save button
    save_button_top_left = (50, y_offset)
    save_button_bottom_right = (250, y_offset + 40)
    cv2.rectangle(sidebar, save_button_top_left, save_button_bottom_right, (0, 128, 0), -1)
    cv2.putText(sidebar, 'Save Invoice', (save_button_top_left[0] + 20, save_button_top_left[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Store the Save button coordinates
    save_button_coords = (save_button_top_left, save_button_bottom_right)

    y_offset += 60

    # Draw the Quit button
    quit_button_top_left = (50, y_offset)
    quit_button_bottom_right = (250, y_offset + 40)
    cv2.rectangle(sidebar, quit_button_top_left, quit_button_bottom_right, (0, 0, 128), -1)
    cv2.putText(sidebar, 'Quit', (quit_button_top_left[0] + 90, quit_button_top_left[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Store the Quit button coordinates
    quit_button_coords = (quit_button_top_left, quit_button_bottom_right)

    combined_frame = np.hstack((frame, sidebar))

    cv2.imshow('Automatic Billing System', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or exit_program:
        break

cap.release()
cv2.destroyAllWindows()
