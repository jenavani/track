from flask import Flask, render_template, url_for, jsonify, session, redirect, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import json

app = Flask(__name__, static_folder='static',  template_folder='templates')
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key in production
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'images', 'profiles')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Mock user class - Replace with your actual user model
class User(UserMixin):
    def __init__(self, id, username, email=None, full_name=None, phone=None, 
                 address=None, emergency_contact=None, profile_image=None, join_date=None):
        self.id = id
        self.username = username
        self.email = email
        self.full_name = full_name
        self.phone = phone
        self.address = address
        self.emergency_contact = emergency_contact
        self.profile_image = profile_image
        self.join_date = join_date or datetime.now()

# Mock database - Replace with your actual database
users = {
    1: User(1, "JohnDoe", "john@example.com", "John Doe", "+1234567890", 
            "123 Main St", "+1987654321", None, datetime(2023, 1, 1))
}

# Mock vehicle data - Replace with your actual vehicle database
vehicle_data = {
    1: {
        "model": "Raptee Electric Motorcycle",
        "registration_number": "TN01AB1234",
        "purchase_date": datetime(2023, 6, 15),
        "battery_health": 95
    }
}

# Mock rewards data
rewards_data = [
    {
        "id": 1,
        "name": "Free Service",
        "points": 1000,
        "description": "Get a free service for your vehicle",
        "image": "service.png"
    },
    {
        "id": 2,
        "name": "Battery Upgrade",
        "points": 5000,
        "description": "Upgrade your battery capacity",
        "image": "battery.png"
    }
]

# Mock insurance data
insurance_plans = {
    1: {
        "type": "Premium Coverage",
        "coverage": 500000,
        "premium": 2999,
        "next_payment": "2024-04-01"
    }
}

@login_manager.user_loader
def load_user(user_id):
    return users.get(int(user_id))

@app.route('/')
def landing():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('landingpage.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Add your user registration logic here
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Mock login - Replace with actual login logic
        user = users[1]
        login_user(user)
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    return render_template('forgot_password.html')

def create_plot():
    # Generate sample driving data
    dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(30, 0, -1)]
    scores = [85 + (i % 10 - 5) for i in range(30)]  # Scores around 85
    
    # Create the main line plot
    fig = go.Figure()
    
    # Add the line trace
    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode='lines+markers',
        name='Driving Score',
        line=dict(color='#2E86C1', width=2),
        marker=dict(size=6)
    ))
    
    # Update layout
    fig.update_layout(
        title='30-Day Driving Score Trend',
        xaxis_title='Date',
        yaxis_title='Score',
        yaxis_range=[60, 100],
        template='plotly_white',
        hovermode='x unified',
        showlegend=False
    )
    
    return pio.to_json(fig)

@app.route('/dashboard')
@login_required
def dashboard():
    plot_json = create_plot()
    return render_template('dashboard.html', 
                         username=current_user.username,
                         plot_json=plot_json)

@app.route('/rewards')
@login_required
def rewards():
    # Mock data - Replace with actual database queries
    total_points = 1500
    redemption_history = [
        {
            "reward_name": "Free Service",
            "reward_image": "service.png",
            "date": "2024-02-15",
            "points": 1000,
            "status": "Completed"
        }
    ]
    return render_template('rewards.html', 
                         total_points=total_points,
                         rewards=rewards_data,
                         redemption_history=redemption_history)

@app.route('/redeem-reward', methods=['POST'])
@login_required
def redeem_reward():
    data = request.json
    reward_id = data.get('reward_id')
    # Add your reward redemption logic here
    return jsonify({"success": True, "message": "Reward redeemed successfully"})

@app.route('/insurance')
@login_required
def insurance():
    # Mock data - Replace with actual database queries
    insurance_status = "Active"
    current_plan = insurance_plans[1]
    coverage_details = [
        {
            "type": "Accident Coverage",
            "description": "Covers all types of accidents",
            "amount": "500000",
            "icon": "accident.png"
        },
        {
            "type": "Third Party Liability",
            "description": "Covers damage to third party",
            "amount": "200000",
            "icon": "third-party.png"
        }
    ]
    claims = [
        {
            "id": "CLM001",
            "date": "2024-01-15",
            "type": "Accident",
            "amount": "25000",
            "status": "Approved"
        }
    ]
    return render_template('insurance.html',
                         insurance_status=insurance_status,
                         current_plan=current_plan,
                         coverage_details=coverage_details,
                         claims=claims)

@app.route('/file-claim')
@login_required
def file_claim():
    return "File Claim Page"  # Create a proper template for this

@app.route('/update-plan')
@login_required
def update_plan():
    return "Update Plan Page"  # Create a proper template for this

@app.route('/profile')
@login_required
def profile():
    # Get vehicle information for the current user
    vehicle = vehicle_data.get(current_user.id, {})
    return render_template('profile.html',
                         user=current_user,
                         vehicle=vehicle)

@app.route('/update-profile', methods=['POST'])
@login_required
def update_profile():
    if request.method == 'POST':
        # Update user information
        current_user.full_name = request.form.get('full_name')
        current_user.email = request.form.get('email')
        current_user.phone = request.form.get('phone')
        current_user.address = request.form.get('address')
        current_user.emergency_contact = request.form.get('emergency_contact')
        # In a real application, save these changes to your database
        flash('Profile updated successfully', 'success')
        return redirect(url_for('profile'))

@app.route('/update-profile-image', methods=['POST'])
@login_required
def update_profile_image():
    if 'profile_image' not in request.files:
        return jsonify({"success": False, "message": "No file provided"})
    
    file = request.files['profile_image']
    if file.filename == '':
        return jsonify({"success": False, "message": "No file selected"})
    
    if file:
        filename = secure_filename(f"user_{current_user.id}_{file.filename}")
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        current_user.profile_image = filename
        # In a real application, save this change to your database
        return jsonify({"success": True, "message": "Profile image updated successfully"})

@app.route('/change-password')
@login_required
def change_password():
    return "Change Password Page"  # Create a proper template for this

@app.route('/notification-settings')
@login_required
def notification_settings():
    return "Notification Settings Page"  # Create a proper template for this

@app.route('/privacy-settings')
@login_required
def privacy_settings():
    return "Privacy Settings Page"  # Create a proper template for this

if __name__ == '__main__':
    app.run(debug=False)
