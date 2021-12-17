from flask import Flask
app = Flask(__name__)

@app.route('/')
@app.route('/test')
def get_index():
    return '<html><center><script>document.write("Hello!")</script></center></html>'