import sqlite3

# Path to the database
db_file = "attendance.db"

# Connect to the database
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Fetch all attendance records
cursor.execute("SELECT * FROM attendance")
rows = cursor.fetchall()

# Display the records
if rows:
    print(f"{'ID':<5} {'Name':<20} {'Timestamp':<20}")
    print("-" * 50)
    for row in rows:
        print(f"{row[0]:<5} {row[1]:<20} {row[2]:<20}")
else:
    print("No attendance records found.")

conn.close()
