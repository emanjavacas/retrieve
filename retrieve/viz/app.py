
import flask
from flask import Flask
from flask_socketio import SocketIO
from flask import render_template

app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app)

@app.route("/")
def index():
    return render_template("index.html")


# request matching for 2 documents
@app.route("/matching", methods=["GET"])
def matching():
    pass


# send request from python to refresh the website
@app.route("/heatmap", methods=["POST"])
def heatmap():
    # get data from request
    data = flask.request.get_json()
    # send heatmap data pushed by python process
    socketio.emit("heatmap", data)
    return {}


@socketio.on("connect")
def io_connect():
    print("Socket client connected")


if __name__ == '__main__':
    socketio.run(app, debug=True)
