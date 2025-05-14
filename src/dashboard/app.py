# Main application file for the dashboard

from . import create_app
import os

# Create the Flask application
app = create_app()

# This block allows us to run the app directly with 'python app.py'
if __name__ == '__main__':
    # Get port from environment variable or use 8080 as default (changed from 5000)
    port = int(os.environ.get('PORT', 8080))
    # Run the application in debug mode (remove debug=True in production)
    app.run(host='0.0.0.0', port=port, debug=True)