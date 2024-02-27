"""
Initialize a sample SQLite database using the Chinook database schema.
"""

import sqlite3


conn = sqlite3.connect("data/chinook.db")
cursor = conn.cursor()

with open("data/Chinook_Sqlite.sql", encoding="utf-8") as f:
    sql = f.read()
    cursor.executescript(sql)

conn.commit()
