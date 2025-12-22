from __future__ import annotations

from pydantic import BaseModel


class UserResponse(BaseModel):
    id: int
    username: str


class TokenResponse(BaseModel):
    token: str


class TokenUserResponse(UserResponse, TokenResponse):
    pass
