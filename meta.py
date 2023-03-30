import os
import cv2

# Function to load label data from a txt file
def load_label_data(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        parts = line.strip().split(' ')
        label = parts[0]
        x_min = int(float(parts[1]) )
        y_min = int(float(parts[2]) )
        x_max = int(float(parts[3]) )
        y_max = int(float(parts[4]) )
        data.append((label, x_min, y_min, x_max, y_max))
    return data

# Function to save label data to a txt file
def save_label_data(label_path, label_data):
    with open(label_path, "w") as f:
        for box in label_data:
            x_min = round(box[1])
            y_min = round(box[2])
            x_max = round(box[3])
            y_max = round(box[4])
            label = box[0]
            f.write(f"{label} {x_min} {y_min} {x_max} {y_max}\n")
# Function to draw rectangles on an image based on label data
def draw_rectangles(img, data):
    for d in data:
        label, x_min, y_min, x_max, y_max = d
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Function to handle mouse events on the image window
# Function to handle mouse events on the image window
def mouse_callback(event, x, y, flags, param):
    global label_data, label_path, drawing_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_rect = True
        label_data.append(("plate", x, y, x, y))  # Add a new rectangle to the label data with initial coordinates
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_rect:
            label_data[-1] = ("plate", label_data[-1][1], label_data[-1][2], x, y)  # Update the last rectangle's coordinates
            img_copy = img.copy()
            draw_rectangles(img_copy, label_data[:-1])  # Draw the previously drawn rectangles
            cv2.rectangle(img_copy, (label_data[-1][1], label_data[-1][2]), (x, y), (0, 255, 0), 2)  # Draw the new rectangle
            cv2.imshow("image", img_copy)  # Show the updated image
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing_rect:
            drawing_rect = False
            draw_rectangles(img, label_data)  # Draw all the rectangles again
            save_label_data(label_path, label_data)  # Save the updated label data to the file
            cv2.imshow("image", img)  # Update the image window

# Main program
img_folder = "C:/Users/srii ideapad/Downloads/yolov5-master/img"
label_folder = "C:/Users/srii ideapad/Downloads/yolov5-master/label"
img_files = os.listdir(img_folder)

window_size = (1080, 960)

# Create a window with a fixed size
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", window_size)

for img_file in img_files:
    img_path = os.path.join(img_folder, img_file)
    label_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + ".txt")

    # Load the image and its corresponding label data (if available)
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape
    if os.path.exists(label_path):
        label_data = load_label_data(label_path)
    else:
        label_data = []

    # Draw the rectangles on the image
    draw_rectangles(img, label_data)

    # Display the image and allow the user to add more rectangles
    cv2.imshow("image", img)
    cv2.setMouseCallback("image", mouse_callback)
    cv2.waitKey(0)

    # Save the label data to the file (if it has changed)
    if len(label_data) > 0:
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        save_label_data(label_path, label_data)
import os
import cv2

# Function to load label data from a txt file
def load_label_data(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        parts = line.strip().split(' ')
        label = int(parts[0])
        x_min = int(float(parts[1]) * img_width)
        y_min = int(float(parts[2]) * img_height)
        x_max = int(float(parts[3]) * img_width)
        y_max = int(float(parts[4]) * img_height)
        data.append((label, x_min, y_min, x_max, y_max))
    return data

# Function to save label data to a txt file
def save_label_data(label_path, data):
    with open(label_path, 'w') as f:
        for d in data:
            label, x_min, y_min, x_max, y_max = d
            f.write(f"{label} {x_min/img_width:.6f} {y_min/img_height:.6f} {x_max/img_width:.6f} {y_max/img_height:.6f}\n")

# Function to draw rectangles on an image based on label data
def draw_rectangles(img, data):
    for d in data:
        label, x_min, y_min, x_max, y_max = d
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Function to handle mouse events on the image window
def mouse_callback(event, x, y, flags, param):
    global label_data, label_path, drawing_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_rect = True
        label_data.append((0, x, y, x, y))  # Add a new rectangle to the label data with initial coordinates
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_rect:
            label_data[-1] = (0, label_data[-1][1], label_data[-1][2], x, y)  # Update the last rectangle's coordinates
            img_copy = img.copy()
            draw_rectangles(img_copy, label_data[:-1])  # Draw the previously drawn rectangles
            cv2.rectangle(img_copy, (label_data[-1][1], label_data[-1][2]), (x, y), (0, 255, 0), 2)  # Draw the new rectangle
            cv2.imshow("image", img_copy)  # Show the updated image
    elif event == cv2.EVENT_LBUTTONUP:
        drawing_rect = False
        draw_rectangles(img, label_data)  # Draw all the rectangles again
        save_label_data(label_path, label_data)  # Save the updated label data to the file
        cv2.imshow("image", img)  # Update the image window

# Main program
img_folder = "images"
label_folder = "labels"
img_files = os.listdir(img_folder)

for img_file in img_files:
    img_path = os.path.join(img_folder, img_file)
    label_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + ".txt")

    # Load the image and its corresponding label data (if available)
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape
    if os.path.exists(label_path):
        label_data = load_label
