# Authentication routes for the dashboard

from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from functools import wraps

# Create a Blueprint for authentication routes
auth = Blueprint('auth', __name__)

# Simple user database (replace with a real database in production)
users = {
    'admin': 'password123'  # NEVER use such simple passwords in production!
}

def login_required(f):
    """Decorator to require login for certain routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

@auth.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('main.index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@auth.route('/logout')
def logout():
    """Logout route"""
    session.pop('user', None)
    return redirect(url_for('auth.login'))