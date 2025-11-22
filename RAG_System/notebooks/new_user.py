from uuid import uuid4
import psycopg2
import json
from datetime import datetime
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
import chainlit as cl
from typing import Optional

# ------------------------------------
identifier = "1"
login = "admin"
password = "1234"
display_name = "Admin"
access = "1"
# ------------------------------------

metadata={"username": login, "password": password, "display_name": display_name, "access": access}

CHAINLIT_CONN = "postgresql+asyncpg://appuser:secret@10.101.10.106:5433/appdb"

@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(
        conninfo=CHAINLIT_CONN
    )

@cl.on_chat_start
async def start_chat():
    #app_user = cl.user_session.get("user")
    cl.user_session.set("chat_history", [])

@cl.password_auth_callback
async def on_login(username: str, password_1: str) -> Optional[cl.User]:                      
    data_layer = get_data_layer()
    user = cl.User(identifier=identifier, metadata=metadata)
    await data_layer.create_user(user)
    print(f"\nПользователь успешно создан.\nlogin: {login}\npassword: {password}\n")
    return cl.User(identifier=identifier, metadata=metadata)

