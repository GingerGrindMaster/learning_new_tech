from flask import Flask, jsonify, request, Response

"""

"""
USERNAME = 'admin'
PASSWORD = '123'

def check_auth(username, password):
    return username == USERNAME and password == PASSWORD

def requires_auth(f):
    """Dekorátor pro vyžadování autorizace."""
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

def authenticate():
    """Vrátí 401 Unauthorized response a žádá o autentizaci."""
    return (jsonify({"error": "Authentication required!"}), 401,
            {'WWW-Authenticate': 'Basic realm="Login Required"'})


app = Flask(__name__)  # create an instance of the Flask class, web-like application
@app.route('/test_success', methods=['GET'])
def test_success():
    return 'Success!'"Úspěšně zabalen Docker Image, následně nahraný na DockerHub", 200

messages = dict()

#format posilani zpravy
#curl -X POST http://localhost:5000/message -H "Content-Type: application/json" -d '{"user_id": "123", "message": "Hello"}'

@app.route('/message', methods=['POST'])
def add_msg():
    data = request.json   # get data from request

    if 'user_id' not in data or 'message' not in data:
        return jsonify({"error": "Invalid POST request! 'user_id' and 'message' are required!"}), 400

    user_id = data['user_id']
    msg = data['message']
    messages[user_id] = msg
    return jsonify({"status" : "message added successfuly", "user_id": user_id, "message": msg}), 200


# curl -X GET http://localhost:5000/messages
#curl -u admin:secret http://localhost:5000/messages

# URL adresa serveru, na kterou se požadavek posílá
@app.route('/messages', methods=['GET'])
@requires_auth  #pri pousteni teto funkce ji obali requires_auth a bude chtit autorizaci

def get_all_msgs():
    return jsonify(messages), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # 0.0.0.0 app pristupna z jakekoliv IP adresy
