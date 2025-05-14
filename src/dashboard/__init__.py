# This file marks the dashboard directory as a Python package
# It allows us to import modules from this directory

# Import necessary modules
from flask import Flask

# Create a function to initialize our Flask application
def create_app():
    # Create a Flask application instance
    app = Flask(__name__)
    
    # Configure basic settings
    app.config['SECRET_KEY'] = 'your-secret-key-for-session'  # Used for session security
    app.config['TEMPLATES_AUTO_RELOAD'] = True  # Auto reload templates during development
    
    # Import and register routes
    from .routes import main_routes, auth_routes
    app.register_blueprint(main_routes.main)
    app.register_blueprint(auth_routes.auth)
    
    return app