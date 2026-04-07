# Smart Surveillance Dashboard
# Professional dashboard with advanced features integration

from flask import Flask
from flask_socketio import SocketIO
import os

# Global SocketIO instance for real-time communication
socketio = SocketIO()

def create_app():
    """Create and configure the Flask application with advanced features."""
    app = Flask(__name__)
    
    # Enhanced configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'smart-surveillance-2025-key')
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file upload
    
    # Initialize SocketIO for real-time updates
    socketio.init_app(app, cors_allowed_origins="*")
    
    # Import and register blueprints
    from .routes import main_routes, auth_routes, api_routes
    app.register_blueprint(main_routes.main)
    app.register_blueprint(auth_routes.auth)
    app.register_blueprint(api_routes.api, url_prefix='/api')
    
    return app