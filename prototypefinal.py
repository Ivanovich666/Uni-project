import cv2
from picamera2 import Picamera2
from datetime import datetime
import time
import os
import shutil
import RPi.GPIO as GPIO

# Set up GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Define GPIO pins
LED_PIN = 17  # GPIO 17 (Pin 11) for LED
BUTTON_PIN = 27  # GPIO 27 (Pin 13) for Button

# Set up the GPIO pins
GPIO.setup(LED_PIN, GPIO.OUT)  # Set LED pin as output
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Set button pin as input with pull-up resistor
GPIO.output(LED_PIN, GPIO.LOW)  # Turn the LED off

source_folder = "/home/Ivanovich/Desktop/Object_Detection_Files/FOTO_STORAGE"
destination_folder = "/home/Ivanovich/Desktop/Object_Detection_Files/captured_images"

save_dir = "captured_images"  # Folder where images are being stored

# Class names and model setup
classNames = []
classFile = "/home/Ivanovich/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/Ivanovich/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/Ivanovich/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

# Initialize the model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize object history to track objects across images
object_history = []

# This is to set up what the drawn box size/color is and the font/size/color of the name tag and confidence label
def getObjects(img, thres, nms, draw=True, objects=[]):
    object_idnumber = 1
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    
    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className, object_idnumber])
                object_idnumber
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=1)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0], box[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0], box[1] - 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    x, y, width, height = box
                    center_x = x + width // 2
                    center_y = y + height // 2
                    if className in ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck']:
                        led_effect()

    return img, objectInfo

def analyze_image(image_path, confidence_threshold=0.45):
    image = cv2.imread(image_path)
    original_image = image.copy()
    original_height, original_width, _ = image.shape

    image_resized = cv2.resize(image, (320, 320))
    class_ids, confidences, boxes = net.detect(image_resized, confThreshold=confidence_threshold)

    objects_data = []

    for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
        x, y, w, h = box

        if class_id - 1 < 0 or class_id - 1 >= len(COCO_CLASSES):
            print(f"Invalid class_id {class_id}. Skipping this object.")
            continue

        x_original = int(x * (original_width / 320))
        y_original = int(y * (original_height / 320))
        w_original = int(w * (original_width / 320))
        h_original = int(h * (original_height / 320))

        cv2.rectangle(original_image, (x_original, y_original), (x_original + w_original, y_original + h_original), (0, 255, 0), 2)

        label = COCO_CLASSES[class_id - 1]
        label_text = f"{label}: {confidence:.2f}"

        cv2.putText(original_image, label_text, (x_original, y_original - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        object_info = {
            "class": label,
            "confidence": confidence,
            "box": (x_original, y_original, x_original + w_original, y_original + h_original)
        }
        objects_data.append(object_info)

    cv2.destroyAllWindows()  # Remove any OpenCV windows opened for preview

    return objects_data, original_image

def analyze_objects_over_time(current_objects_data, current_image):
    global object_history

    # Define a list of classes to highlight
    target_classes = ['car', 'train', 'person', 'truck', 'bicycle']

    # Ensure we store the current object data for comparison
    if len(object_history) >= 6:
        object_history.pop(0)  # Remove the oldest record

    object_history.append(current_objects_data)

    # Check if we have 6 images to compare
    if len(object_history) == 6:
        # Initialize a counter to check how many objects are in similar locations
        similar_location_count = 0

        for i in range(1, 6):  # Iterate over the previous 5 images
            current_objects = current_objects_data
            previous_objects = object_history[i]

            # Compare the current image's objects with the previous ones
            for current_object in current_objects:
                for prev_object in previous_objects:
                    if current_object["class"] == prev_object["class"]:
                        # Compare if the object location is similar (within a threshold)
                        x1, y1, x2, y2 = current_object["box"]
                        px1, py1, px2, py2 = prev_object["box"]
                        # Check if the object has moved very little (within 20px)
                        if abs(x1 - px1) < 20 and abs(y1 - py1) < 20:
                            similar_location_count += 1

        # If 6 consecutive images show the object in similar locations, highlight them in red
        if similar_location_count >= 6:
            print("Detected consistent object(s) over time!")
            led_effect()

            # Highlight these objects in red
            for obj in current_objects_data:
                label = obj["class"]
                if label in target_classes:  # Only highlight if the class is in the target list
                    x1, y1, x2, y2 = obj["box"]
                    # Draw the bounding box in red color on the original image
                    cv2.rectangle(current_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color for highlight
                    # Optionally, display the label as well
                    label_text = f"{label}: {obj['confidence']:.2f}"
                    cv2.putText(current_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Show the image in its original size (without resizing)
            cv2.imshow("Analyzed Image with Red Highlights", current_image)
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyAllWindows()  # Close the OpenCV window after key press
        else:
            print("Objects are not consistent over time.")

def cam_take_picture(picam, source_folder):
    if not os.path.exists(source_folder):
        os.makedirs(source_folder)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.jpg"
    filepath = os.path.join(save_dir, filename)

    picam.start()  
    time.sleep(5)  # Give time for the camera to warm up

    print("Picture capture in progress...")

    # Capture the image without preview
    picam.capture_file(filepath)  # Capture the image
    picam.stop()

    print(f"Picture saved: {filepath}")

    return filepath

def led_effect():
    try:
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(4)
        GPIO.output(LED_PIN, GPIO.LOW)
        time.sleep(1)
    except KeyboardInterrupt:
        GPIO.cleanup()
        print("Program exited cleanly.")

def move_file_testing(source_folder, destination_folder):
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return None

    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    if not files:
        print("No files left to move.")
        return None

    files.sort(key=lambda f: os.path.getmtime(os.path.join(source_folder, f)))

    file_name = files[0]
    source_path = os.path.join(source_folder, file_name)
    destination_path = os.path.join(destination_folder, file_name)

    try:
        shutil.move(source_path, destination_path)
        print(f"Moved: {file_name} -> {destination_folder}")
        return file_name
    except Exception as e:
        print(f"Error moving {file_name}: {e}")
        return None

def main():
    picam = Picamera2()

    print("Waiting for button press to start...")
    while GPIO.input(BUTTON_PIN):
        time.sleep(0.1)

    try:
        while True:
            # Take a picture only in main function
            picture_path = cam_take_picture(picam, source_folder)

            # Analyze the image after capture
            objects_data, current_image = analyze_image(picture_path)

            # Analyze objects over time
            analyze_objects_over_time(objects_data, current_image)

            time.sleep(5)  # Wait for a while before taking another picture
    except KeyboardInterrupt:
        GPIO.cleanup()
        print("Program exited cleanly.")

if __name__ == "__main__":
    main()



