"""Every error leaves the gateway as {"error": {"code", "message", "type"}}: OpenAI's shape, so OpenAI SDKs show
the message, and never a provider's text, account, region or request id."""

import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

log = logging.getLogger("ai-gateway")


class GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}
        super().__init__(message)


def unauthorized():
    return GatewayError(401, "UNAUTHORIZED", "Missing or invalid DeployKar AI key.",
                        {"WWW-Authenticate": 'Bearer realm="deploykar-ai"'})


def error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message, "type": code.lower()}}


async def gateway_error_handler(_: Request, exc: GatewayError):
    return JSONResponse(error_body(exc.code, exc.message), status_code=exc.status, headers=exc.headers)


async def validation_error_handler(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(p) for p in first.get("loc", []) if p not in ("body", "query"))
    message = f"{where}: {first.get('msg', 'invalid value')}" if where else first.get("msg", "Invalid request.")
    return JSONResponse(error_body("INVALID_REQUEST", message), status_code=400)


async def unexpected_error_handler(_: Request, exc: Exception):
    log.exception("unhandled error", exc_info=exc)
    return JSONResponse(error_body("INTERNAL_ERROR", "Something went wrong. Please retry."), status_code=500)
