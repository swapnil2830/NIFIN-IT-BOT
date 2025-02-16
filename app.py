import copy
import json
import os
import logging
import uuid
import httpx
import asyncio
import jwt
from jwt import PyJWKClient

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

# Environment variables for Google Chat validation
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID") or "YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com"
# If your Chat token has iss=chat@system.gserviceaccount.com, set that as default or override via env
GOOGLE_ISSUER = os.environ.get("GOOGLE_ISSUER", "chat@system.gserviceaccount.com")
GOOGLE_JWKS_URI = "https://www.googleapis.com/oauth2/v3/certs"

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

app = create_app()

# Debug settings
DEBUG = os.environ.get("DEBUG", "false")
if DEBUG.lower() == "true":
    logging.basicConfig(level=logging.DEBUG)

USER_AGENT = "GitHubSampleWebApp/AsyncAzureOpenAI/1.0.0"

frontend_settings = {
    "auth_enabled": app_settings.base_settings.auth_enabled,
    "feedback_enabled": (
        app_settings.chat_history and
        app_settings.chat_history.enable_feedback
    ),
    "ui": {
        "title": app_settings.ui.title,
        "logo": app_settings.ui.logo,
        "chat_logo": app_settings.ui.chat_logo or app_settings.ui.logo,
        "chat_title": app_settings.ui.chat_title,
        "chat_description": app_settings.ui.chat_description,
        "show_share_button": app_settings.ui.show_share_button,
        "show_chat_history_button": app_settings.ui.show_chat_history_button,
    },
    "sanitize_answer": app_settings.base_settings.sanitize_answer,
    "oyd_enabled": app_settings.base_settings.datasource_type,
}

# Enable Microsoft Defender for Cloud
MS_DEFENDER_ENABLED = os.environ.get("MS_DEFENDER_ENABLED", "true").lower() == "true"

@bp.route("/")
async def index():
    return await render_template(
        "index.html",
        title=app_settings.ui.title,
        favicon=app_settings.ui.favicon
    )

@bp.route("/favicon.ico")
async def favicon():
    return await bp.send_static_file("favicon.ico")

@bp.route("/assets/<path:path>")
async def assets(path):
    return await send_from_directory("static/assets", path)

@bp.route("/webhook", methods=["POST"])
async def google_chat_webhook():
    """
    Google Chat Webhook with JWT Validation
    """
    try:
        # Get Auth header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            logging.warning("Missing or invalid Authorization header")
            return jsonify({"error": "Unauthorized"}), 401

        token = auth_header.split("Bearer ")[-1]

        # Validate JWT from Google Chat
        decoded_token = validate_google_jwt(token)
        if not decoded_token:
            return jsonify({"error": "Invalid token"}), 401

        request_json = await request.get_json()
        event_type = request_json.get("type", "")

        if event_type == "MESSAGE":
            user_message = request_json.get("message", {}).get("text", "")
            user_name = request_json.get("message", {}).get("sender", {}).get("displayName", "User")
            logging.info(f"User '{user_name}' sent message: {user_message}")

            response_text = await handle_google_chat_message(user_message, user_name)
            return jsonify({"text": response_text}), 200

        elif event_type in ("ADDED_TO_SPACE", "REMOVED_FROM_SPACE", "CARD_CLICKED"):
            logging.info(f"Handling event type '{event_type}' from Chat.")
            return jsonify({"text": f"Handled event type: {event_type}"}), 200

        else:
            logging.info(f"Unknown event type: {event_type}")
            return jsonify({"text": "Unknown event type."}), 200

    except Exception as e:
        logging.exception("Error handling Google Chat webhook")
        return jsonify({"error": str(e)}), 500

def validate_google_jwt(token: str):
    """
    Validate Google Chat-signed JWT using PyJWKClient.
    If the token is valid, return the decoded claims. Otherwise, return None.
    """
    try:
        jwk_client = PyJWKClient(GOOGLE_JWKS_URI)
        signing_key = jwk_client.get_signing_key_from_jwt(token)

        decoded_token = jwt.decode(
            token,
            key=signing_key.key,
            algorithms=["RS256"],
            audience=GOOGLE_CLIENT_ID,
            issuer=GOOGLE_ISSUER,
            options={"verify_exp": True}
        )
        return decoded_token
    except jwt.ExpiredSignatureError:
        logging.error("Google Chat JWT token expired.")
        return None
    except jwt.InvalidTokenError as ex:
        logging.error(f"Invalid token error: {ex}")
        return None

async def handle_google_chat_message(user_message, user_name):
    """
    Process the user's message and return an Azure OpenAI-based response.
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
    """
    Initialize Azure OpenAI client with env settings
    """
    try:
        endpoint = app_settings.azure_openai.endpoint or f"https://{app_settings.azure_openai.resource}.openai.azure.com/"
        aoai_api_key = app_settings.azure_openai.key
        # If there's no API key, you might use Azure AD token-based auth instead.

        return AsyncAzureOpenAI(
            api_version=app_settings.azure_openai.preview_api_version,
            api_key=aoai_api_key,
            azure_endpoint=endpoint,
        )
    except Exception as e:
        logging.exception("Exception initializing Azure OpenAI client", e)
        return None

async def init_cosmosdb_client():
    """
    Initialize Cosmos DB client for conversation history
    """
    cosmos_conversation_client = None
    if app_settings.chat_history:
        try:
            cosmos_endpoint = f"https://{app_settings.chat_history.account}.documents.azure.com:443/"

            if not app_settings.chat_history.account_key:
                async with DefaultAzureCredential() as cred:
                    credential = cred
            else:
                credential = app_settings.chat_history.account_key

            cosmos_conversation_client = CosmosConversationClient(
                cosmosdb_endpoint=cosmos_endpoint,
                credential=credential,
                database_name=app_settings.chat_history.database,
                container_name=app_settings.chat_history.conversations_container,
                enable_message_feedback=app_settings.chat_history.enable_feedback,
            )
        except Exception as e:
            logging.exception("Exception in CosmosDB initialization", e)
            cosmos_conversation_client = None
            raise e
    else:
        logging.debug("CosmosDB not configured")

    return cosmos_conversation_client

def prepare_model_args(request_body, request_headers):
    """
    Utility to build chat request arguments for Azure OpenAI.
    """
    request_messages = request_body.get("messages", [])
    messages = []
    if not app_settings.datasource:
        # system prompt if desired
        messages = [
            {
                "role": "system",
                "content": app_settings.azure_openai.system_message
            }
        ]

    for message in request_messages:
        if message:
            if message["role"] == "assistant" and "context" in message:
                context_obj = json.loads(message["context"])
                messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"],
                        "context": context_obj
                    }
                )
            else:
                messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"]
                    }
                )

    user_json = None
    if MS_DEFENDER_ENABLED:
        authenticated_user_details = get_authenticated_user_details(request_headers)
        conversation_id = request_body.get("conversation_id", None)
        application_name = app_settings.ui.title
        user_json = get_msdefender_user_json(
            authenticated_user_details,
            request_headers,
            conversation_id,
            application_name
        )

    model_args = {
        "messages": messages,
        "temperature": app_settings.azure_openai.temperature,
        "max_tokens": app_settings.azure_openai.max_tokens,
        "top_p": app_settings.azure_openai.top_p,
        "stop": app_settings.azure_openai.stop_sequence,
        "stream": app_settings.azure_openai.stream,
        "model": app_settings.azure_openai.model,
        "user": user_json
    }

    # For retrieval augmentation or data source usage
    if app_settings.datasource:
        model_args["extra_body"] = {
            "data_sources": [
                app_settings.datasource.construct_payload_configuration(request=request)
            ]
        }

    # Redacted logs
    model_args_clean = copy.deepcopy(model_args)
    if model_args_clean.get("extra_body"):
        secret_params = [
            "key",
            "connection_string",
            "embedding_key",
            "encoded_api_key",
            "api_key",
        ]
        # Hide secrets in logs
        ds_params = model_args_clean["extra_body"]["data_sources"][0]["parameters"]
        for secret_param in secret_params:
            if ds_params.get(secret_param):
                ds_params[secret_param] = "*****"
        auth_obj = ds_params.get("authentication", {})
        for field in auth_obj:
            if field in secret_params:
                auth_obj[field] = "*****"
        embed_dep = ds_params.get("embedding_dependency", {})
        if "authentication" in embed_dep:
            for field in embed_dep["authentication"]:
                if field in secret_params:
                    embed_dep["authentication"][field] = "*****"

    logging.debug(f"REQUEST BODY: {json.dumps(model_args_clean, indent=4)}")
    return model_args

async def promptflow_request(request_data):
    """
    Example promptflow request
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {app_settings.promptflow.api_key}",
        }
        async with httpx.AsyncClient(timeout=float(app_settings.promptflow.response_timeout)) as client:
            pf_formatted = convert_to_pf_format(
                request_data,
                app_settings.promptflow.request_field_name,
                app_settings.promptflow.response_field_name
            )
            response = await client.post(
                app_settings.promptflow.endpoint,
                json={
                    app_settings.promptflow.request_field_name: pf_formatted[-1]["inputs"][app_settings.promptflow.request_field_name],
                    "chat_history": pf_formatted[:-1],
                },
                headers=headers,
            )
        resp = response.json()
        resp["id"] = request_data["messages"][-1]["id"]
        return resp
    except Exception as e:
        logging.error(f"An error occurred in promptflow_request: {e}")

async def send_chat_request(request_body, request_headers):
    """
    Send user messages to Azure OpenAI, filter out 'tool' role messages, log request, return response
    """
    filtered_messages = []
    messages = request_body.get("messages", [])
    for message in messages:
        if message.get("role") != 'tool':
            filtered_messages.append(message)
    request_body['messages'] = filtered_messages

    model_args = prepare_model_args(request_body, request_headers)

    try:
        azure_openai_client = await init_openai_client()
        raw_response = await azure_openai_client.chat.completions.with_raw_response.create(**model_args)

        logging.info(f"OpenAI API Request: {json.dumps(model_args, indent=4)}")
        logging.info(f"Authentication Headers: {dict(request_headers)}")

        response = raw_response.parse()
        apim_request_id = raw_response.headers.get("apim-request-id")
    except Exception as e:
        logging.exception("Exception in send_chat_request")
        raise e

    return response, apim_request_id

async def complete_chat_request(request_body, request_headers):
    if app_settings.base_settings.use_promptflow:
        response = await promptflow_request(request_body)
        history_metadata = request_body.get("history_metadata", {})
        return format_pf_non_streaming_response(
            response,
            history_metadata,
            app_settings.promptflow.response_field_name,
            app_settings.promptflow.citations_field_name
        )
    else:
        response, apim_request_id = await send_chat_request(request_body, request_headers)
        history_metadata = request_body.get("history_metadata", {})
        return format_non_streaming_response(response, history_metadata, apim_request_id)

async def stream_chat_request(request_body, request_headers):
    response, apim_request_id = await send_chat_request(request_body, request_headers)
    history_metadata = request_body.get("history_metadata", {})

    async def generate():
        async for completionChunk in response:
            yield format_stream_response(completionChunk, history_metadata, apim_request_id)

    return generate()

async def conversation_internal(request_body, request_headers):
    """
    Common handler for chat conversations
    """
    try:
        if app_settings.azure_openai.stream and not app_settings.base_settings.use_promptflow:
            result = await stream_chat_request(request_body, request_headers)
            response = await make_response(format_as_ndjson(result))
            response.timeout = None
            response.mimetype = "application/json-lines"
            return response
        else:
            result = await complete_chat_request(request_body, request_headers)
            return jsonify(result)

    except Exception as ex:
        logging.exception(ex)
        if hasattr(ex, "status_code"):
            return jsonify({"error": str(ex)}), ex.status_code
        else:
            return jsonify({"error": str(ex)}), 500

@bp.route("/conversation", methods=["POST"])
async def conversation():
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()
    return await conversation_internal(request_json, request.headers)

@bp.route("/frontend_settings", methods=["GET"])
def get_frontend_settings():
    try:
        return jsonify(frontend_settings), 200
    except Exception as e:
        logging.exception("Exception in /frontend_settings")
        return jsonify({"error": str(e)}), 500

########################################
#            HISTORY ENDPOINTS         #
########################################

@bp.route("/history/generate", methods=["POST"])
async def add_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        history_metadata = {}
        if not conversation_id:
            title = await generate_title(request_json["messages"])
            conversation_dict = await current_app.cosmos_conversation_client.create_conversation(
                user_id=user_id, title=title
            )
            conversation_id = conversation_dict["id"]
            history_metadata["title"] = title
            history_metadata["date"] = conversation_dict["createdAt"]

        messages = request_json["messages"]
        if len(messages) > 0 and messages[-1]["role"] == "user":
            created_msg_val = await current_app.cosmos_conversation_client.create_message(
                uuid=str(uuid.uuid4()),
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1],
            )
            if created_msg_val == "Conversation not found":
                raise Exception(
