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
    session,
    redirect,
    url_for
)
from authlib.integrations.starlette_client import OAuth
from openai import AsyncAzureOpenAI
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from backend.auth.auth_utils import get_authenticated_user_details
from backend.security.ms_defender_utils import get_msdefender_user_json
from backend.history.cosmosdbservice import CosmosConversationClient
from backend.settings import app_settings, MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
from backend.utils import (
    format_as_ndjson,
    format_stream_response,
    format_non_streaming_response,
    convert_to_pf_format,
    format_pf_non_streaming_response,
)

bp = Blueprint("routes", __name__, static_folder="static", template_folder="static")
cosmos_db_ready = asyncio.Event()

# OAuth Setup
oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    authorize_params={"scope": "openid email profile"},
    access_token_url="https://oauth2.googleapis.com/token",
    client_kwargs={"scope": "openid email profile"},
    redirect_uri=os.getenv("GOOGLE_REDIRECT_URI"),
)


def create_app():
    app = Quart(__name__)
    app.register_blueprint(bp)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.secret_key = os.getenv("APP_SECRET_KEY", "your_secret_key")
    
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


# ✅ Google OAuth Login
@bp.route("/login")
async def login():
    return await oauth.google.authorize_redirect(url_for("routes.auth_callback", _external=True))


# ✅ Google OAuth Callback (Handles Authentication)
@bp.route("/oauth/callback")
async def auth_callback():
    try:
        token = await oauth.google.authorize_access_token()
        user_info = await oauth.google.parse_id_token(token)
        session["user"] = user_info
        return redirect(url_for("routes.profile"))
    except Exception as e:
        logging.exception("Google OAuth Error")
        return jsonify({"error": "Authentication failed", "details": str(e)}), 500


# ✅ User Profile Endpoint (Checks Login)
@bp.route("/user")
async def profile():
    user = session.get("user")
    if user:
        return jsonify(user)
    return jsonify({"error": "User not logged in"}), 401


# ✅ Logout
@bp.route("/logout")
async def logout():
    session.pop("user", None)
    return redirect(url_for("routes.index"))


@bp.route("/webhook", methods=["POST"])
async def google_chat_webhook():
    try:
        request_json = await request.get_json()
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
            return jsonify({})
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
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.exception("Error in Azure OpenAI response")
        return "Sorry, I couldn't process your message."


async def init_openai_client():
    try:
        endpoint = app_settings.azure_openai.endpoint or f"https://{app_settings.azure_openai.resource}.openai.azure.com/"
        aoai_api_key = app_settings.azure_openai.key
        ad_token_provider = None
        if not aoai_api_key:
            async with DefaultAzureCredential() as credential:
                ad_token_provider = get_bearer_token_provider(
                    credential,
                    "https://cognitiveservices.azure.com/.default"
                )
        azure_openai_client = AsyncAzureOpenAI(
            api_version=app_settings.azure_openai.preview_api_version,
            api_key=aoai_api_key,
            azure_ad_token_provider=ad_token_provider,
            default_headers={"x-ms-useragent": "GitHubSampleWebApp/AsyncAzureOpenAI/1.0.0"},
            azure_endpoint=endpoint,
        )
        return azure_openai_client
    except Exception as e:
        logging.exception("Exception in Azure OpenAI initialization", e)
        raise e


async def init_cosmosdb_client():
    if app_settings.chat_history:
        try:
            cosmos_endpoint = f"https://{app_settings.chat_history.account}.documents.azure.com:443/"
            credential = (
                DefaultAzureCredential()
                if not app_settings.chat_history.account_key
                else app_settings.chat_history.account_key
            )
            return CosmosConversationClient(
                cosmosdb_endpoint=cosmos_endpoint,
                credential=credential,
                database_name=app_settings.chat_history.database,
                container_name=app_settings.chat_history.conversations_container,
                enable_message_feedback=app_settings.chat_history.enable_feedback,
            )
        except Exception as e:
            logging.exception("Exception in CosmosDB initialization", e)
            raise e
    return None


app = create_app()
