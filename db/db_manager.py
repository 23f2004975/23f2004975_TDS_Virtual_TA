from dotenv import load_dotenv
from logging_config import logger
import os
import sqlite3
import traceback
from fastapi import  HTTPException


load_dotenv()
DB_PATH = os.getenv("DB_PATH")

def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    except sqlite3.Error as e:
        error_msg = f"Database connection error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)