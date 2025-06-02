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
    app = Flask(__name__, static_folder='static')
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
    plant_detection_model = YOLO("models\\plant.pt")

    model = YOLO("models\\bittergourd.pt")

    tom_model= YOLO("models\\tomato.pt")


    # Bitter Gourd categories
    bitter_gourd_diseases = {
        'DM': 'Downy Mildew (Disease)',
        'LS': 'Leaf Spot (Disease)',
        'JAS': 'Jassid (Insect)',
        'K': 'Potassium Deficiency (Nutritional)',
        'K Mg': 'Potassium and Magnesium Deficiency (Nutritional)',
        'N': 'Nitrogen Deficiency (Nutritional)',
        'N K': 'Nitrogen and Potassium Deficiency (Nutritional)',
        'N Mg': 'Nitrogen and Magnesium Deficiency (Nutritional)',
        'Healthy': 'Healthy'
    }

    # Tomato categories
    tomato_diseases = {
        'LM': 'Leaf Miner (Insect)',
        'MIT': 'Mite (Insect)',
        'JAS MIT': 'Jassid and Mite (Insect)',
        'K': 'Potassium Deficiency (Nutritional)',
        'N': 'Nitrogen Deficiency (Nutritional)',
        'N K': 'Nitrogen and Potassium Deficiency (Nutritional)',
        'Healthy': 'Healthy'
    }

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
            
            # Check if username or email already exists
            existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
            if existing_user:
                if existing_user.username == username:
                    flash('Username already exists. Please choose a different one.', 'danger')
                elif existing_user.email == email:
                    flash('Email already exists. Please use a different email address.', 'danger')
                return redirect(url_for('signup'))
            
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(username=username, email=email, password=hashed_password)
            db.session.add(user)
            db.session.commit()
            flash('Account created successfully! You can now log in.', 'success')
            return redirect(url_for('login'))
        
        return render_template('signup.html')



    # Default landing page (Welcome page)
    @app.route('/')
    def welcome():
        return render_template('welcome.html')

    # Home route after login
    @app.route('/home')
    @login_required
    def home():
        return render_template('home.html')
    

    # Add this route to handle the redirect after selecting a plant
    @app.route('/detect_redirect', methods=['POST'])
    def detect_redirect():
        plant = request.form.get('plant')
        
        if plant == 'bittergourd':
            return redirect(url_for('predict', plant='bittergourd'))
        elif plant == 'tomato':
            return redirect(url_for('predict', plant='tomato'))
        
        return redirect(url_for('home'))


    
    # Predict route
    @app.route('/predict/<plant>', methods=['GET', 'POST'])
    @login_required
    def predict(plant):
        if 'file' not in request.files:
            return render_template('detect.html', plant=plant, prediction_text="No file provided.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('detect.html', plant=plant, prediction_text="No image selected for uploading.")
        
        try:
            # Create the uploads directory if it doesn't exist
            uploads_dir = os.path.join("static", "uploads")
            if not os.path.exists(uploads_dir):
                os.makedirs(uploads_dir)

            img = Image.open(file)
            img_path = os.path.join(uploads_dir, file.filename)
            img.save(img_path)  # Save the image in static/uploads for processing

            # Step 1: Use the InferenceHTTPClient API to check the class of the image (Runs for both plants)
            try:
                result = CLIENT.infer(img_path, model_id="obstacle-detection-yeuzf/5")
                api_predictions = result.get('predictions', [])

                # Check if any class is detected
                if not api_predictions:
                    # If no predictions from the API (empty), proceed
                    print("No predictions from the API. Proceeding with plant detection.")
                else:
                    # Process API predictions
                    api_class = api_predictions[0].get('class', '')
                    if api_class not in ['tree', 'leaf', 'plant']:
                        # If the API predicts something other than plant-related, stop and show a message
                        return render_template(
                            'detect.html', plant=plant,
                            prediction_text=f"The image contains a {api_class}. Plant Cure only works on plant images."
                        )
            except Exception as api_error:
                # If API call fails or raises an exception, log it and proceed
                print(f"API Error: {api_error}. Proceeding with plant detection.")

            # Step 2: Use the plant detection model only for bitter gourd
            if plant == 'bittergourd':
                plant_results = plant_detection_model.predict(source=img)
                plant_present = False

                for result in plant_results:
                    for box in result.boxes:
                        confidence = round(box.conf[0].item(), 2)
                        if confidence > 0.60:
                            plant_present = True
                            break
                
                if not plant_present:
                    # Save the result image in static/uploads
                    img_filename = "uploaded_image.png"
                    img_path = os.path.join(uploads_dir, img_filename)
                    img.save(img_path)

                    img_url = url_for('static', filename=f"uploads/{img_filename}")
                    return render_template(
                        'detect.html',
                        plant=plant,
                        prediction_text="The picture does not contain a plant or is unclear. Plant Cure works best on plants.",
                        img_url=img_url,
                    )
                

            # Step 3: Detect diseases using the specific model for Bitter Gourd or Tomato
            if plant == 'bittergourd':
                detection_model = model  # Bitter Gourd model
                plant_diseases = bitter_gourd_diseases  # Bitter Gourd diseases mapping
                
                # Adjust confidence threshold to 75% for Bitter Gourd
                confidence_threshold = 0.75
            elif plant == 'tomato':
                detection_model = tom_model  # Tomato model
                plant_diseases = tomato_diseases  # Tomato diseases mapping
                
                # Adjust confidence threshold to 70% for Tomato
                confidence_threshold = 0.70
            else:
                return render_template('detect.html', plant=plant, prediction_text="Invalid plant selection.")

            results = detection_model.predict(source=img)
            predictions = []
            draw = ImageDraw.Draw(img)
            has_valid_prediction = False
            predicted_class = None
            show_info_button = False

            for result in results:
                for box in result.boxes:
                    class_name = result.names[box.cls[0].item()]
                    confidence = round(box.conf[0].item(), 2)
                    bbox = [round(coord, 2) for coord in box.xyxy[0].tolist()]
                    
                    if confidence >= confidence_threshold:
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

            # Save the result image in static/uploads
            img_filename = "detected_image.png"
            img_path = os.path.join(uploads_dir, img_filename)
            img.save(img_path)
            
            img_url = url_for('static', filename=f"uploads/{img_filename}")

            # Pass the show_info_button flag and predicted_class to the template
            return render_template('detect.html', prediction_text=prediction_text, plant=plant, img_url=img_url, predicted_class=predicted_class, show_info_button=show_info_button)

        except Exception as e:
            print(f"Error processing image: {e}")
            return render_template('detect.html', plant=plant, prediction_text="Error processing image.")






    # Modify the info route to receive the predicted class
    @app.route('/info/<plant>/<predicted_class>', methods=['GET', 'POST'])
    @login_required
    def info(plant, predicted_class):
        return render_template('info.html', plant=plant, predicted_class=predicted_class)




    

    



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
