from typing import Any, Dict, Mapping, Optional, Union, cast

import jwt as PyJWT
from fastapi import Depends
from fastapi.exceptions import HTTPException
from fastapi.security.utils import get_authorization_scheme_param
from pydantic import BaseModel
from starlette.requests import Request
from starlette.status import HTTP_401_UNAUTHORIZED
from fastapi.security.base import SecurityBase
from fastapi.openapi.models import OAuth2 as OAuth2Model, OAuthFlows as OAuthFlowsModel

from app.configuration import (
    JWT_ALGORITHM,
    JWT_SECRET,
    JWT_USER_EMAIL_FIELD_NAME,
    JWT_USER_ID_FIELD_NAME,
    JWT_USER_NAME_FIELD_NAME,
)


class JWTBase(SecurityBase):
    def __init__(self, *, scheme_name: str = None):
        flows: OAuthFlowsModel = OAuthFlowsModel(password={"tokenUrl": "/api/users/login"})
        self.model = OAuth2Model(flows=flows)
        self.scheme_name = scheme_name or self.__class__.__name__

    async def __call__(self, request: Request) -> Optional[str]:
        authorization: str = request.headers.get("Authorization")
        if not authorization:
            if "access_token" not in request.query_params:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                param = request.query_params["access_token"]
        else:
            scheme, param = get_authorization_scheme_param(authorization)
            if scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return param


class JWTUser(BaseModel):
    id: str
    role: str
    username: str
    email: str


jwt_token_scheme = JWTBase()


def encode_token(payload, **kwargs) -> str:
    try:
        token = PyJWT.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM).decode(
            encoding="UTF-8"
        )
        return token
    except Exception as exc:
        print(exc.__class__.__name__)
        raise


def decode_token(token: str) -> Optional[JWTUser]:
    try:
        payload = PyJWT.decode(token, JWT_SECRET, algorithms=JWT_ALGORITHM)
        if payload == {}:
            return None
    except PyJWT.MissingRequiredClaimError as exc:
        # print(f'JWT Missing claim: {exc.claim}')
        return None
    except PyJWT.InvalidTokenError as exc:
        print(f"JWT Invalid Token: {exc.__class__.__name__}")
        return None
    except Exception as exc:
        print(f"JWT Exception: {exc.__class__.__name__}")
        return None
    _id = payload.get(JWT_USER_ID_FIELD_NAME)
    username = payload.get(JWT_USER_NAME_FIELD_NAME)
    email = payload.get(JWT_USER_EMAIL_FIELD_NAME)
    role = payload.get("role")
    return JWTUser(id=_id, username=username, email=email, role=role)


async def get_current_user(token: str = Depends(jwt_token_scheme)):
    user = decode_token(token)
    if user is None:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED, detail="Bad authentication"
        )
    return user
