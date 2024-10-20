import os
from flask import Flask, request, render_template, redirect, url_for, flash
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from ultralytics import YOLO
from PIL import Image
from extensions import db, login_manager  # Import db and login_manager from extensions
from flask import send_file
import io
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient


# Initialize the Flask app
def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your_secret_key'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

    db.init_app(app)
    login_manager.init_app(app)
    bcrypt = Bcrypt(app)

    CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="HNggSix27PGd4G84jAHN"  # Your API key
)



    from Database import User  # Import models here after initializing the app to avoid circular imports

    with app.app_context():
        # Automatically create the database if it doesn't exist
        if not os.path.exists('site.db'):
            db.create_all()
            print('Database created successfully.')

    # Load the YOLOv8 model
    # Load the new model for plant detection (YOLOv8 or any other)
    plant_detection_model = YOLO("C:\\Users\\ASUS\\Downloads\\Research New\\Codes and Data\\test\\runs\\detect\\plant detection\\weights\\best.pt")

    model = YOLO("C:\\Users\\ASUS\\Downloads\\Research New\\Codes and Data\\test\\runs\\detect\\train30\\weights\\best.pt")

    # Default landing page (Welcome page)
    @app.route('/')
    def welcome():
        return render_template('welcome.html')

    # Home route after login
    @app.route('/home')
    @login_required
    def home():
        return render_template('home.html')


    # Predict route
    @app.route('/predict', methods=['GET', 'POST'])
    @login_required
    def predict():
        if 'file' not in request.files:
            return render_template('detect.html', prediction_text="No file provided.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('detect.html', prediction_text="No image selected for uploading.")
        
        try:
            img = Image.open(file)
            img_path = os.path.join("static", file.filename)
            img.save(img_path)  # Save the image temporarily for API processing

            # Step 1: Use the InferenceHTTPClient API to check the class of the image
            try:
                result = CLIENT.infer(img_path, model_id="obstacle-detection-yeuzf/5")
                api_predictions = result.get('predictions', [])

                # Check if any class is detected
                if not api_predictions:
                    # If no predictions from the API (empty), proceed to plant detection
                    print("No predictions from the API. Proceeding with plant detection.")
                else:
                    # Process API predictions
                    api_class = api_predictions[0].get('class', '')
                    if api_class not in ['tree', 'leaf', 'plant']:
                        # If the API predicts something other than plant-related, stop and show a message
                        return render_template(
                            'detect.html',
                            prediction_text=f"The image contains a {api_class}. Plant Cure only works on plant images."
                        )
            except Exception as api_error:
                # If API call fails or raises an exception, log it and proceed to plant detection
                print(f"API Error: {api_error}. Proceeding with plant detection.")

            # Step 2: Use the plant detection model (YOLOv8) even if the API returns no results or predicts 'tree', 'leaf', or 'plant'
            plant_results = plant_detection_model.predict(source=img)
            plant_present = False

            # Check if any bounding box with high confidence is detected
            for result in plant_results:
                for box in result.boxes:
                    confidence = round(box.conf[0].item(), 2)
                    if confidence > 0.50:
                        plant_present = True
                        break
            
            if not plant_present:
                # If no plant is detected, return a message
                return render_template(
                    'detect.html',
                    prediction_text="The picture does not contain a plant or is unclear. Plant Cure works best on plants."
                )
            
            # Step 3: Proceed with the disease detection model (if plant detection is successful)
            results = model.predict(source=img)
            predictions = []
            draw = ImageDraw.Draw(img)
            has_valid_prediction = False
            predicted_class = None  # Store the predicted class
            show_info_button = False  # This flag controls the visibility of the "Know More" button

            for result in results:
                for box in result.boxes:
                    class_name = result.names[box.cls[0].item()]
                    confidence = round(box.conf[0].item(), 2)
                    bbox = [round(coord, 2) for coord in box.xyxy[0].tolist()]
                    
                    if confidence > 0.90:
                        predictions.append({
                            'class': class_name,
                            'confidence': confidence,
                            'box': bbox
                        })
                        has_valid_prediction = True
                        predicted_class = class_name  # Capture the predicted class
                        show_info_button = True  # Set flag to true if prediction is valid

                        # Draw the bounding box on the image
                        draw.rectangle(bbox, outline="red", width=3)
                        draw.text((bbox[0], bbox[1]), f"{class_name} {confidence}", fill="red")
            
            if has_valid_prediction:
                prediction_text = "Predictions:\n"
                for pred in predictions:
                    prediction_text += f"Class: {pred['class']}, Confidence: {pred['confidence']}, Box: {pred['box']}\n"
            else:
                prediction_text = "Unidentified Disease or image not clear. Please upload a better quality image."
            
            img_filename = "detected_image.png"
            img_path = os.path.join("static", img_filename)
            img.save(img_path)
            
            img_url = url_for('static', filename=img_filename)

            # Pass the show_info_button flag and predicted_class to the template
            return render_template('detect.html', prediction_text=prediction_text, img_url=img_url, predicted_class=predicted_class, show_info_button=show_info_button)
        
        except Exception as e:
            return render_template('detect.html', prediction_text=f"Error processing image: {e}")






    # Modify the info route to receive the predicted class
    @app.route('/info/<condition>', methods=['GET', 'POST'])
    @login_required
    def info(condition):
        return render_template('info.html', condition=condition)



    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('home'))
        if request.method == 'POST':
            email = request.form['email']
            password = request.form['password']
            user = User.query.filter_by(email=email).first()
            if user and bcrypt.check_password_hash(user.password, password):
                login_user(user)
                flash('Login successful!', 'success')
                return redirect(url_for('home'))  # Redirect to home after successful login
            else:
                flash('Login failed. Check email and password.', 'danger')
        return render_template('login.html')

    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        if current_user.is_authenticated:
            return redirect(url_for('home'))
        if request.method == 'POST':
            username = request.form['username']
            email = request.form['email']
            password = request.form['password']
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(username=username, email=email, password=hashed_password)
            db.session.add(user)
            db.session.commit()
            flash('Account created successfully! You can now log in.', 'success')
            return redirect(url_for('login'))
        return render_template('signup.html')

    

    # Add this route to handle the redirect after selecting a plant
    @app.route('/detect_redirect', methods=['POST'])
    def detect_redirect():
        plant = request.form['plant']
        
        if plant == 'bittergourd':
            return redirect(url_for('predict'))
        
        return redirect(url_for('home'))

    # Add this route to handle plant suggestions
    @app.route('/suggest_plant', methods=['POST'])
    @login_required
    def suggest_plant():
        plant_suggestion = request.form.get('plant_suggestion')
        
        if plant_suggestion:
            # You can handle the suggestion here, e.g., save it to a database, send an email, etc.
            flash(f'Thank you for suggesting: {plant_suggestion}. We will review it.', 'success')
        else:
            flash('Please provide a valid plant name.', 'danger')
        
        return redirect(url_for('home'))



    # Logout route
    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        return redirect(url_for('home'))


    # Admin page (for admin users only)
    @app.route('/admin')
    @login_required
    def admin():
        if not current_user.is_admin:
            flash('Admin access only.', 'danger')
            return redirect(url_for('home'))
        return render_template('admin.html')

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
