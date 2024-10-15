import os
from flask import Flask, request, render_template, redirect, url_for, flash
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from ultralytics import YOLO
from PIL import Image
from extensions import db, login_manager  # Import db and login_manager from extensions

# Initialize the Flask app
def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your_secret_key'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

    db.init_app(app)
    login_manager.init_app(app)
    bcrypt = Bcrypt(app)


    from Database import User  # Import models here after initializing the app to avoid circular imports

    with app.app_context():
        # Automatically create the database if it doesn't exist
        if not os.path.exists('site.db'):
            db.create_all()
            print('Database created successfully.')

    # Load the YOLOv8 model
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
            results = model.predict(source=img)
            
            predictions = []
            for result in results:
                for box in result.boxes:
                    predictions.append({
                        'class': result.names[box.cls[0].item()],
                        'confidence': round(box.conf[0].item(), 2),
                        'box': [round(coord, 2) for coord in box.xyxy[0].tolist()]
                    })
            
            if predictions:
                prediction_text = "Predictions:\n"
                for pred in predictions:
                    prediction_text += f"Class: {pred['class']}, Confidence: {pred['confidence']}, Box: {pred['box']}\n"
            else:
                prediction_text = "No objects detected."
            
            return render_template('detect.html', prediction_text=prediction_text)
        
        except Exception as e:
            return render_template('detect.html', prediction_text=f"Error processing image: {e}")


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
