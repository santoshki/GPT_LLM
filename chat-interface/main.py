from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from entities import invoke_api

app = Flask(__name__)
CORS(app)

# -------------------------
# UI Route
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------
# Chat API
# -------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    print("User:", user_message)

    try:
        # bot_response = invoke_llm.generate_model_response(user_message)
        bot_response = invoke_api.generate_response(user_message)
    except Exception as e:
        print("Error:", e)
        bot_response = None

    if not isinstance(bot_response, str) or not bot_response.strip():
        bot_response = "I'm not sure I understand. Could you explain more?"

    print("Bot:", bot_response)

    return jsonify({
        "response": bot_response
    })


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)