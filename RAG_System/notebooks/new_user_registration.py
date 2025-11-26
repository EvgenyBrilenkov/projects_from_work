import psycopg2
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
import chainlit as cl
from typing import Optional

DB_CONN = "dbname=appdb user=appuser password=secret port=5432 host=10.101.10.106"
CHAINLIT_CONN = "postgresql+asyncpg://appuser:secret@10.101.10.106:5432/appdb"

# -----------NEW-USER-DATA------------
login = "test"
password = "1234"
display_name = "Test"
access = "2"
# ------------------------------------

conn = psycopg2.connect(DB_CONN)
cur = conn.cursor()

cur.execute(
            """
            SELECT MAX(identifier)
            FROM users
            """
            )

identifier = str(int(cur.fetchall()[0][0]) + 1)

cur.close()
conn.close()

metadata={"username": login, "password": password, "display_name": display_name, "access": access}

@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(
        conninfo=CHAINLIT_CONN
    )

@cl.on_chat_start
async def start_chat():
    pass

@cl.password_auth_callback
async def on_login(username: str, password_1: str) -> Optional[cl.User]:                      
    data_layer = get_data_layer()
    user = cl.User(identifier=identifier, metadata=metadata)
    await data_layer.create_user(user)
    print(f"\nПользователь успешно создан.\nlogin: {login}\npassword: {password}\nid: {identifier}\n")
    return cl.User(identifier=identifier, metadata=metadata)

