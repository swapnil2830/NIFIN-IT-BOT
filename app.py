from quart import Quart, jsonify, request
from google.auth import jwt
import asyncio
import logging
import os

# Import your existing functions and settings
from backend.settings import app_settings
from backend.utils import prepare_model_args, send_chat_request

app = Quart(__name__)

# Configure logging
DEBUG = os.environ.get("DEBUG", "false")
if DEBUG.lower() == "true":
    logging.basicConfig(level=logging.DEBUG)

# Constants for Google Chat verification
GOOGLE_CHAT_CERTS_URL = "https://www.googleapis.com/service_accounts/v1/metadata/x509/chat@system.gserviceaccount.com"
GOOGLE_CHAT_CLIENT_ID = "185257238474-vqo558j115fpunbi2kooqc8nv675q2rv.apps.googleusercontent.com"  # Replace with your Google Chat bot client ID

# Initialize your existing Azure OpenAI Client or any dependencies here
async def init_openai_client():
    # Example placeholder for initializing your Azure OpenAI client
    return None


# Verify requests from Google Chat
def verify_google_chat_request(request):
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return False

    token = auth_header.split("Bearer ")[-1]
    try:
        info = jwt.decode(token, certs_url=GOOGLE_CHAT_CERTS_URL)
        return info["iss"] == "chat@system.gserviceaccount.com"
    except Exception as e:
        logging.error(f"Failed to verify Google Chat request: {e}")
        return False


# Google Chat webhook endpoint
@app.route("/google-chat", methods=["POST"])
async def google_chat_webhook():
    # Verify that the request is from Google Chat
    if not verify_google_chat_request(request):
        return jsonify({"error": "Unauthorized"}), 401

    # Parse the incoming request from Google Chat
    event = await request.json
    user_message = event.get("message", {}).get("text", "Hello")

    # Process the user message with Azure OpenAI
    response_text = await process_gpt_message(user_message)

    # Respond back to Google Chat
    return jsonify({"text": response_text})


# Process messages using Azure OpenAI
async def process_gpt_message(user_message):
    try:
        # Prepare model arguments
        model_args = prepare_model_args(
            {
                "messages": [{"role": "user", "content": user_message}],
            },
            {}
        )

        # Send the message to Azure OpenAI
        response, _ = await send_chat_request(model_args, {})
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"Failed to process GPT message: {e}")
        return "Sorry, something went wrong."


# Main app setup
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
