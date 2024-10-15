import requests

# API endpoint for image prediction
url = 'http://localhost:5000/predict_api'

# Open the image file in binary mode to send via POST request
image_path = 'C:\\Users\\ASUS\\Downloads\\Research New\\sample_image.jpg'
with open(image_path, 'rb') as image_file:
    files = {'file': image_file}
    
    # Send a POST request with the image file
    r = requests.post(url, files=files)

# Print the API response (should contain YOLOv8 prediction results)
print(r.json())
