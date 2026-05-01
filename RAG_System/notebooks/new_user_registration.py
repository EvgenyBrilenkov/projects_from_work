import psycopg2
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
import chainlit as cl
from typing import Optional

DB_CONN = "dbname=appdb user=appuser password=secret port=5432 host=db"
CHAINLIT_CONN = "postgresql+asyncpg://appuser:secret@db:5432/appdb"

# -----------NEW-USER-DATA------------
login = "admin"
password = "admin"
display_name = "Admin"
access = "1"
# ------------------------------------

conn = psycopg2.connect(DB_CONN)
cur = conn.cursor()

cur.execute(
            """
            SELECT MAX(identifier)
            FROM users
            """
            )

a = cur.fetchall()

if a[0][0]:
    identifier = str(int(a[0][0]) + 1)
else:
    identifier = '0'

cur.close()
conn.close()

metadata={"username": login, "password": password, "display_name": display_name, "access": access}


@cl.password_auth_callback
async def on_login(username: str, password_1: str) -> Optional[cl.User]:    
    print('login')                  
    data_layer = get_data_layer()
    user = cl.User(identifier=identifier, metadata=metadata)
    await data_layer.create_user(user)
    print(f"\nПользователь успешно создан.\nlogin: {login}\npassword: {password}\nid: {identifier}\n")
    return cl.User(identifier=identifier, metadata=metadata)

@cl.data_layer
def get_data_layer():
    print("Initializing data layer...")
    return SQLAlchemyDataLayer(
        conninfo=CHAINLIT_CONN
    )

@cl.on_chat_start
async def start_chat():
    pass