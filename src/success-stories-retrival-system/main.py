from server import mcp
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from utils.auth import extract_token_from_headers, verify_jwt_token
from utils.request_context import request_var
from utils.mongodb_singleton import get_mongodb_client
from utils.session_history_manager import SessionHistoryManager
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route
from llm.azurecustomllm import AzureCustomLLM
from download_response import download_message as convert_message , download_cleanup
import uvicorn
from typing import Optional
import os

import asyncio
from starlette.responses import JSONResponse, StreamingResponse, PlainTextResponse, Response


# Initialize MongoDB singleton
mongo_client = get_mongodb_client()

# Initialize session manager with dependency injection
session_manager = SessionHistoryManager(mongo_client)

import tools.SSchatbot

# custom_middleware = [
#     Middleware(CORSMiddleware, allow_origins=["*"],
#             allow_credentials=True,
#             allow_methods=["*"],
#             allow_headers=["*"],)
# ]

# Auth Middleware
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_var.set(request)
        
        if request.method.upper() == "OPTIONS":
            return await call_next(request)
        
        # Handle GET requests to /mcp endpoint (health checks)
        # if request.url.path.startswith("/mcp") and request.method.upper() == "GET":
        #     return JSONResponse(
        #         status_code=200,
        #         content={
        #             "status": "ok",
        #             "version": "1.0.0",
        #             "endpoints": {
        #                 "mcp": "/mcp",
        #                 "methods": ["GET", "POST", "OPTIONS"]
        #             }
        #         }
        #     )

        if request.url.path.startswith("/mcp") and request.method.upper() == "GET":
            accept = (request.headers.get("accept") or "").lower()
            if "text/event-stream" in accept:
                # SSE client (EventSource) -> pass through to FastMCP SSE
                return await call_next(request)
            else:
                # Non-SSE caller -> return a simple status JSON
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "ok",
                        "version": "1.0.0",
                        "endpoints": {"mcp": "/mcp", "methods": ["GET", "POST", "OPTIONS"]},
                    },
                    headers={
                        # Make it explicit this is not an SSE stream
                        "Cache-Control": "no-cache",
                    },
                )

        if request.url.path.startswith("/mcp") and request.method.upper() == "POST":
            try:
                body_bytes = await request.body()
                payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
            except Exception:
                payload = {}

            tool_name = (
                payload.get("name")
                or (payload.get("params") or {}).get("name")
                or payload.get("tool")
                or payload.get("operation")
            )

            if not tool_name:
                return await call_next(request)

            if tool_name in ["login_user", "refresh_jwt_token"]:
                return await call_next(request)

            token = extract_token_from_headers(dict(request.headers))
            if not token:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "OAuthError",
                        "message": "Missing authentication token in headers",
                    },
                )

            try:
                claims = verify_jwt_token(token)
                request.state.jwt_claims = claims
            except Exception as e:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "OAuthError",
                        "message": f"Invalid or expired token: {str(e)}",
                    },
                )

        return await call_next(request)


class SecurityAndCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data: https:; object-src 'none'; frame-ancestors 'none';"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "Authorization, Content-Type, Set-Cookie"
        # Remove server header to hide server technology (VAPT requirement)
        if "Server" in response.headers:
            del response.headers["Server"]
        return response

custom_middleware = [
    Middleware(AuthMiddleware),
    Middleware(SecurityAndCORSMiddleware),
]


# Create the MCP HTTP app
base_app = mcp.http_app(
    transport="http",
    path="/mcp",
    middleware=custom_middleware,
    stateless_http=True
)

async def mcp_get(request: Request):
    accept = (request.headers.get("accept") or "").lower()

    if "text/event-stream" in accept:
        # Lightweight SSE stream to keep connection open and avoid reconnect storm
        async def stream():
            try:
                # Open the stream
                yield b": connected\n\n"
                while True:
                    # 1) If client closed the connection, stop the loop
                    if await request.is_disconnected():
                        break
                    
                    # 2) Send a heartbeat every 15s
                    yield b": keep-alive\n\n"
                    await asyncio.sleep(15)
            
            except asyncio.CancelledError:
                # 3) Uvicorn is shutting down (Ctrl+C) — exit promptly
                # (do any cleanup you need here)
                pass



        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "version": "1.0.0", "endpoints": {"mcp": "/mcp", "methods": ["GET", "POST", "OPTIONS"]}},
        headers={"Cache-Control": "no-cache"},
    )

base_app.add_route("/mcp", mcp_get, methods=["GET"])

async def download_message(request: Request):
    data = await request.json()
    message = data.get("message", "")
    doc_format = data.get("format","md")
    llm = AzureCustomLLM()
   
    refined_message = llm.invoke(sys_prompt = download_cleanup['sys_prompt'] ,input = message)
   
    docx_stream = await convert_message(refined_message,doc_format)
    docx_stream.seek(0)
    return StreamingResponse(
        docx_stream,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=message.docx"}
    )
    
# Register the route with the app
base_app.add_route("/download_message", download_message, methods=["POST"])

# Cookie Wrapper
class CookieWrapperApp:
    """Wrapper to intercept responses and set HttpOnly cookies"""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False
        status_code = None
        headers = []
        body_parts = []

        async def send_wrapper(message):
            nonlocal response_started, status_code, headers, body_parts

            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                headers = list(message.get("headers", []))
                return

            elif message["type"] == "http.response.body":
                body_parts.append(message.get("body", b""))
                if not message.get("more_body", False):
                    full_body = b"".join(body_parts)
                    try:
                        request = request_var.get()
                        if request and hasattr(request.state, 'refresh_token') and request.state.refresh_token:
                            refresh_token = request.state.refresh_token
                            refresh_token_expires = getattr(request.state, 'refresh_token_expires', 86400)
                            cookie_value = f"refresh_token={refresh_token}; Path=/; Max-Age={refresh_token_expires}; HttpOnly secure=true"
                            headers.append((b"set-cookie", cookie_value.encode("utf-8")))
                    except Exception:
                        pass

                    # Remove server header to hide server technology (VAPT requirement)
                    headers = [(name, value) for name, value in headers if name.lower() != b"server"]
                    
                    await send({
                        "type": "http.response.start",
                        "status": status_code,
                        "headers": headers,
                    })
                    await send({
                        "type": "http.response.body",
                        "body": full_body,
                    })
            else:
                await send(message)

        await self.app(scope, receive, send_wrapper)


# Wrap the MCP app
http_app = CookieWrapperApp(base_app)