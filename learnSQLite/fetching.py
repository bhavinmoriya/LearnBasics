import sqlite3

conn = sqlite3.connect(":memory:")   # in-memory DB for demo
cur = conn.cursor()

# Create demo table
cur.execute("CREATE TABLE users (id INTEGER, name TEXT)")
cur.execute("INSERT INTO users VALUES (?, ?)", (1, "Alice"))
cur.execute("INSERT INTO users VALUES (?, ?)", (2, "Bob"))
conn.commit()

# Query
cur.execute("SELECT * FROM users")

print(cur.fetchone())  # (1, 'Alice')
print(cur.fetchone())  # (2, 'Bob')
print(cur.fetchone())  # None (no more rows)

cur.execute("SELECT * FROM users")
rows = cur.fetchall()
print(rows)  # [(1, 'Alice'), (2, 'Bob')]

cur.execute("SELECT * FROM users")
print(cur.fetchmany(1))  # [(1, 'Alice')]
print(cur.fetchmany(1))  # [(2, 'Bob')]
print(cur.fetchmany(1))  # []

cur.execute("SELECT * FROM users")
for row in cur:
    print(row)  # each row is a tuple
