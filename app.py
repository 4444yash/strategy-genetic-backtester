from flask import Flask, render_template, send_from_directory
from api.routes_universe import universe_bp
from api.routes_pool import pool_bp
from api.routes_run import run_bp
import os
import webbrowser
from threading import Timer

app = Flask(__name__, static_folder='static', template_folder='.')

# Register Blueprints
app.register_blueprint(universe_bp)
app.register_blueprint(pool_bp)
app.register_blueprint(run_bp)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    # Timer(1.5, open_browser).start()
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
