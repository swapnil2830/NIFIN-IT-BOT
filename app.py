from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# Replace with your external API endpoint or logic
EXTERNAL_API_URL = "https://ns-AZAI-svc.openai.azure.com/openai/deployments/gpt-4o-mini/completions?api-version=2023-03-15"
response = requests.post(
    EXTERNAL_API_URL,
    headers={
        "Authorization": f"Bearer {os.getenv('4QfGXH6AG2PHuSHwcieFhvmKgCIVmlZD0WqkzlPJYViEvN5TWEA2JQQJ99ALACHrzpqXJ3w3AAAAACOG1Iss')}",
        "Content-Type": "application/json",
    },
    json={
        "prompt": message_text,
        "max_tokens": 100,
    }
)


@app.route('/google-chat-webhook', methods=['POST'])
def google_chat_webhook():
    """
    This endpoint handles incoming messages from Google Chat.
    """
    data = request.get_json()

    # Extract message and user details from Google Chat payload
    message_text = data.get('message', {}).get('text', '')
    user_name = data.get('user', {}).get('displayName', 'User')

    # Prepare response message
    if message_text.lower() == 'hello':
        reply_text = f"Hello, {user_name}! How can I assist you today?"
    elif message_text.lower() == 'help':
        reply_text = "Here are some commands you can use: \n- `Hello`: Greet the bot\n- `Help`: Get help"
    else:
        # Forward the message to your external API for processing
        external_api_response = requests.post(
            EXTERNAL_API_URL,
            json={"message": message_text}
        )
        if external_api_response.status_code == 200:
            reply_text = external_api_response.json().get('result', 'Sorry, I couldn’t process your request.')
        else:
            reply_text = "Sorry, there was an error processing your request."

    # Create response for Google Chat
    response = {
        "text": reply_text
    }

    return jsonify(response), 200


if __name__ == '__main__':
    # Run the app on port 8080, which is required for Google Cloud Run
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
