# Importing required libraries and setting environment
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load YOLOv8 model (change path if using a custom model)
model = YOLO("C:\\Users\\ASUS\\Downloads\\Research New\\Codes and Data\\test\\runs\\detect\\train30\\weights\\best.pt")  # Use your model path here


# Load a trained model (either pre-trained or custom-trained)
# Replace "yolov8n.pt" with your custom model if needed
#model = YOLO("yolov8n.pt")

# Predicting on a new image
# Specify the path to your image
# image_path = "C:\\Users\\ASUS\\Downloads\\Research New\\sample_image.jpg"

# # Run prediction using the YOLOv8 model
# results = model.predict(source=image_path, save=True)

# # Displaying results
# # Assuming that results come with bounding boxes or detection data
# # Here we display the image with detections if available
# result_image = results[0].plot()  # Plotting the first result

# # Show the image with predictions
# plt.imshow(result_image)
# plt.axis('off')
# plt.show()

# # Save the detection result image (optional)
# output_image_path = "C:\\Users\\ASUS\\Downloads\\Research New\\prediction_output.jpg"
# plt.imsave(output_image_path, result_image)

# # Print prediction details (bounding boxes, classes, confidence scores, etc.)
# for result in results:
#     print(f"Classes detected: {result.names}")
#     print(f"Bounding boxes: {result.boxes}")
#     print(f"Confidence scores: {result.conf}")
