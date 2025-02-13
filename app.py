import copy
import json
import os
import logging
import uuid
import httpx
import asyncio
from quart import (
    Blueprint,
    Quart,
    jsonify,
    make_response,
    request,
    send_from_directory,
    render_template,
    current_app,
)

from openai import AsyncAzureOpenAI
from azure.identity.aio import (
    DefaultAzureCredential,
    get_bearer_token_provider
)
from backend.auth.auth_utils import get_authenticated_user_details
from backend.security.ms_defender_utils import get_msdefender_user_json
from backend.history.cosmosdbservice import CosmosConversationClient
from backend.settings import (
    app_settings,
    MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
)
from backend.utils import (
    format_as_ndjson,
    format_stream_response,
    format_non_streaming_response,
    convert_to_pf_format,
    format_pf_non_streaming_response,
)

bp = Blueprint("routes", __name__, static_folder="static", template_folder="static")

cosmos_db_ready = asyncio.Event()


def create_app():
    app = Quart(__name__)
    app.register_blueprint(bp)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    
    @app.before_serving
    async def init():
        try:
            app.cosmos_conversation_client = await init_cosmosdb_client()
            cosmos_db_ready.set()
        except Exception as e:
            logging.exception("Failed to initialize CosmosDB client")
            app.cosmos_conversation_client = None
            raise e
    
    return app


@bp.route("/")
async def index():
    return await render_template(
        "index.html",
        title=app_settings.ui.title,
        favicon=app_settings.ui.favicon
    )


@bp.route("/webhook", methods=["POST"])
async def google_chat_webhook():
    try:
        request_json = await request.get_json()
        logging.info(f"Google Chat Webhook Request: {json.dumps(request_json, indent=4)}")

        # Authenticate Google Chat request
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            logging.warning("Missing Authorization Header")
            return jsonify({"error": "Unauthorized"}), 401

        event_type = request_json.get("type")

        if event_type == "MESSAGE":
            user_message = request_json.get("message", {}).get("text", "")
            user_name = request_json.get("message", {}).get("sender", {}).get("displayName", "User")
            response_text = await handle_google_chat_message(user_message, user_name)
            return jsonify({"text": response_text})

        elif event_type == "ADDED_TO_SPACE":
            space_name = request_json.get("space", {}).get("name", "unknown space")
            return jsonify({"text": f"Thanks for adding me to {space_name}!"})

        elif event_type == "REMOVED_FROM_SPACE":
            return jsonify({})  # Handle bot removal logic if necessary

        else:
            return jsonify({"text": "I didn't understand that event type."})

    except Exception as e:
        logging.exception("Error handling Google Chat webhook")
        return jsonify({"error": str(e)}), 500


async def handle_google_chat_message(user_message, user_name):
    """
    Process the user's message and return a response.
    This function integrates with Azure OpenAI.
    """
    try:
        azure_openai_client = await init_openai_client()
        response = await azure_openai_client.chat.completions.create(
            model=app_settings.azure_openai.model,
            messages=[
                {"role": "system", "content": "You are a helpful IT assistant."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.exception("Error in Azure OpenAI response")
        return "Sorry, I couldn't process your message."


# Initialize Azure OpenAI Client
async def init_openai_client():
    try:
        if (
            app_settings.azure_openai.preview_api_version
            < MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
        ):
            raise ValueError(
                f"Minimum supported Azure OpenAI preview API version is '{MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION}'"
            )

        endpoint = (
            app_settings.azure_openai.endpoint
            if app_settings.azure_openai.endpoint
            else f"https://{app_settings.azure_openai.resource}.openai.azure.com/"
        )

        aoai_api_key = app_settings.azure_openai.key
        ad_token_provider = None

        if not aoai_api_key:
            logging.debug("No AZURE_OPENAI_KEY found, using Azure Entra ID auth")
            async with DefaultAzureCredential() as credential:
                ad_token_provider = get_bearer_token_provider(
                    credential,
                    "https://cognitiveservices.azure.com/.default"
                )

        return AsyncAzureOpenAI(
            api_version=app_settings.azure_openai.preview_api_version,
            api_key=aoai_api_key,
            azure_ad_token_provider=ad_token_provider,
            default_headers={"x-ms-useragent": "GitHubSampleWebApp/AsyncAzureOpenAI/1.0.0"},
            azure_endpoint=endpoint,
        )

    except Exception as e:
        logging.exception("Exception in Azure OpenAI initialization")
        raise e


async def init_cosmosdb_client():
    if not app_settings.chat_history:
        logging.debug("CosmosDB not configured")
        return None

    try:
        cosmos_endpoint = f"https://{app_settings.chat_history.account}.documents.azure.com:443/"

        credential = (
            app_settings.chat_history.account_key
            if app_settings.chat_history.account_key
            else DefaultAzureCredential()
        )

        return CosmosConversationClient(
            cosmosdb_endpoint=cosmos_endpoint,
            credential=credential,
            database_name=app_settings.chat_history.database,
            container_name=app_settings.chat_history.conversations_container,
            enable_message_feedback=app_settings.chat_history.enable_feedback,
        )

    except Exception as e:
        logging.exception("Exception in CosmosDB initialization")
        raise e


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
