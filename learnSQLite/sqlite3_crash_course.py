#!/usr/bin/env python3
"""
SQLite3 Python Crash Course
Complete guide with practical examples
"""

import sqlite3
import pandas as pd
from datetime import datetime

# =============================================================================
# 1. BASIC CONNECTION AND DATABASE CREATION
# =============================================================================

# Connect to database (creates file if doesn't exist)
conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# In-memory database (temporary, lost when connection closes)
# conn = sqlite3.connect(':memory:')

print("1. DATABASE CONNECTION")
print("Connected to SQLite database")

# =============================================================================
# 2. CREATE TABLES
# =============================================================================

print("\n2. CREATE TABLES")

# Create a simple table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER,
        department TEXT,
        salary REAL,
        hire_date TEXT
    )
''')

# Create another table with foreign key
cursor.execute('''
    CREATE TABLE IF NOT EXISTS projects (
        project_id INTEGER PRIMARY KEY,
        project_name TEXT NOT NULL,
        employee_id INTEGER,
        budget REAL,
        FOREIGN KEY (employee_id) REFERENCES employees (id)
    )
''')

conn.commit()
print("Tables created successfully")

# =============================================================================
# 3. INSERT DATA
# =============================================================================

print("\n3. INSERT DATA")

# Insert single record
cursor.execute('''
    INSERT INTO employees (name, age, department, salary, hire_date)
    VALUES (?, ?, ?, ?, ?)
''', ('Alice Johnson', 30, 'Engineering', 75000.0, '2023-01-15'))

# Insert multiple records
employees_data = [
    ('Bob Smith', 25, 'Marketing', 55000.0, '2023-03-20'),
    ('Charlie Brown', 35, 'Engineering', 85000.0, '2022-11-10'),
    ('Diana Wilson', 28, 'HR', 60000.0, '2023-02-05'),
    ('Eve Davis', 32, 'Engineering', 80000.0, '2022-08-15')
]

cursor.executemany('''
    INSERT INTO employees (name, age, department, salary, hire_date)
    VALUES (?, ?, ?, ?, ?)
''', employees_data)

# Insert projects
projects_data = [
    (1, 'Website Redesign', 1, 50000.0),
    (2, 'Mobile App', 3, 120000.0),
    (3, 'Data Analytics', 5, 80000.0),
    (4, 'Marketing Campaign', 2, 30000.0)
]

cursor.executemany('''
    INSERT INTO projects (project_id, project_name, employee_id, budget)
    VALUES (?, ?, ?, ?)
''', projects_data)

conn.commit()
print(f"Inserted {len(employees_data) + 1} employees and {len(projects_data)} projects")

# =============================================================================
# 4. SELECT QUERIES (READ DATA)
# =============================================================================

print("\n4. SELECT QUERIES")

# Select all records
print("\n4.1 All employees:")
cursor.execute('SELECT * FROM employees')
all_employees = cursor.fetchall()
for emp in all_employees:
    print(f"  {emp}")

# Select specific columns
print("\n4.2 Names and salaries:")
cursor.execute('SELECT name, salary FROM employees')
names_salaries = cursor.fetchall()
for name, salary in names_salaries:
    print(f"  {name}: ${salary:,.2f}")

# Select with WHERE clause
print("\n4.3 Engineering employees:")
cursor.execute('SELECT name, age FROM employees WHERE department = ?', ('Engineering',))
eng_employees = cursor.fetchall()
for name, age in eng_employees:
    print(f"  {name} (age {age})")

# Select with ORDER BY
print("\n4.4 Employees by salary (highest first):")
cursor.execute('SELECT name, salary FROM employees ORDER BY salary DESC')
sorted_employees = cursor.fetchall()
for name, salary in sorted_employees:
    print(f"  {name}: ${salary:,.2f}")

# Select with LIMIT
print("\n4.5 Top 2 highest paid employees:")
cursor.execute('SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 2')
top_employees = cursor.fetchall()
for name, salary in top_employees:
    print(f"  {name}: ${salary:,.2f}")

# =============================================================================
# 5. AGGREGATE FUNCTIONS
# =============================================================================

print("\n5. AGGREGATE FUNCTIONS")

# Count records
cursor.execute('SELECT COUNT(*) FROM employees')
count = cursor.fetchone()[0]
print(f"Total employees: {count}")

# Average salary
cursor.execute('SELECT AVG(salary) FROM employees')
avg_salary = cursor.fetchone()[0]
print(f"Average salary: ${avg_salary:,.2f}")

# Group by department
print("\nEmployees by department:")
cursor.execute('''
    SELECT department, COUNT(*) as count, AVG(salary) as avg_salary
    FROM employees 
    GROUP BY department
    ORDER BY avg_salary DESC
''')
dept_stats = cursor.fetchall()
for dept, count, avg_sal in dept_stats:
    print(f"  {dept}: {count} employees, avg salary ${avg_sal:,.2f}")

# =============================================================================
# 6. JOINS
# =============================================================================

print("\n6. JOINS")

print("Employees with their projects:")
cursor.execute('''
    SELECT e.name, p.project_name, p.budget
    FROM employees e
    LEFT JOIN projects p ON e.id = p.employee_id
    ORDER BY e.name
''')
emp_projects = cursor.fetchall()
for name, project, budget in emp_projects:
    project_info = f"{project} (${budget:,.2f})" if project else "No project assigned"
    print(f"  {name}: {project_info}")

# =============================================================================
# 7. UPDATE RECORDS
# =============================================================================

print("\n7. UPDATE RECORDS")

# Update single record
cursor.execute('''
    UPDATE employees 
    SET salary = salary * 1.1 
    WHERE name = ?
''', ('Alice Johnson',))

# Update multiple records
cursor.execute('''
    UPDATE employees 
    SET salary = salary * 1.05 
    WHERE department = ?
''', ('Engineering',))

conn.commit()

print("Updated salaries:")
cursor.execute('SELECT name, salary FROM employees ORDER BY salary DESC')
updated_salaries = cursor.fetchall()
for name, salary in updated_salaries:
    print(f"  {name}: ${salary:,.2f}")

# =============================================================================
# 8. DELETE RECORDS
# =============================================================================

print("\n8. DELETE RECORDS")

# Delete with condition
cursor.execute('DELETE FROM projects WHERE budget < ?', (40000,))
deleted_count = cursor.rowcount
conn.commit()

print(f"Deleted {deleted_count} projects with budget < $40,000")

# =============================================================================
# 9. TRANSACTIONS
# =============================================================================

print("\n9. TRANSACTIONS")

try:
    # Start transaction (implicit)
    cursor.execute('INSERT INTO employees (name, age, department, salary, hire_date) VALUES (?, ?, ?, ?, ?)',
                   ('Frank Miller', 29, 'Finance', 65000.0, '2023-09-01'))
    
    cursor.execute('INSERT INTO projects (project_id, project_name, employee_id, budget) VALUES (?, ?, ?, ?)',
                   (5, 'Budget Analysis', cursor.lastrowid, 45000.0))
    
    # Commit transaction
    conn.commit()
    print("Transaction completed successfully")
    
except sqlite3.Error as e:
    # Rollback on error
    conn.rollback()
    print(f"Transaction failed: {e}")

# =============================================================================
# 10. WORKING WITH PANDAS
# =============================================================================

print("\n10. PANDAS INTEGRATION")

# Read data into pandas DataFrame
df_employees = pd.read_sql_query('SELECT * FROM employees', conn)
print("Employees DataFrame:")
print(df_employees.head())

# Write pandas DataFrame to SQLite
new_data = pd.DataFrame({
    'name': ['Grace Lee', 'Henry Kim'],
    'age': [27, 31],
    'department': ['Design', 'Engineering'],
    'salary': [58000.0, 78000.0],
    'hire_date': ['2023-04-10', '2023-05-22']
})

new_data.to_sql('employees', conn, if_exists='append', index=False)
print(f"\nAdded {len(new_data)} records from DataFrame")

# =============================================================================
# 11. ADVANCED QUERIES
# =============================================================================

print("\n11. ADVANCED QUERIES")

# Subqueries
print("Employees with above-average salary:")
cursor.execute('''
    SELECT name, salary 
    FROM employees 
    WHERE salary > (SELECT AVG(salary) FROM employees)
    ORDER BY salary DESC
''')
above_avg = cursor.fetchall()
for name, salary in above_avg:
    print(f"  {name}: ${salary:,.2f}")

# Window functions (SQLite 3.25+)
print("\nSalary ranking:")
cursor.execute('''
    SELECT name, salary,
           RANK() OVER (ORDER BY salary DESC) as rank
    FROM employees
    ORDER BY rank
''')
rankings = cursor.fetchall()
for name, salary, rank in rankings:
    print(f"  #{rank}: {name} - ${salary:,.2f}")

# =============================================================================
# 12. DATABASE METADATA
# =============================================================================

print("\n12. DATABASE METADATA")

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables in database:")
for table in tables:
    print(f"  - {table[0]}")

# Get table schema
cursor.execute("PRAGMA table_info(employees)")
columns = cursor.fetchall()
print("\nEmployees table schema:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")

# =============================================================================
# 13. ERROR HANDLING
# =============================================================================

print("\n13. ERROR HANDLING")

try:
    cursor.execute("SELECT * FROM non_existent_table")
except sqlite3.OperationalError as e:
    print(f"Caught error: {e}")

try:
    cursor.execute("INSERT INTO employees (name) VALUES (?)", (None,))
except sqlite3.IntegrityError as e:
    print(f"Constraint violation: {e}")

# =============================================================================
# 14. BEST PRACTICES EXAMPLES
# =============================================================================

print("\n14. BEST PRACTICES")

def get_employee_by_id(emp_id):
    """Safe function with proper error handling"""
    try:
        cursor.execute('SELECT * FROM employees WHERE id = ?', (emp_id,))
        result = cursor.fetchone()
        if result:
            return {
                'id': result[0], 'name': result[1], 'age': result[2],
                'department': result[3], 'salary': result[4], 'hire_date': result[5]
            }
        return None
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None

# Context manager for automatic cleanup
def safe_database_operation():
    """Using context manager for safe database operations"""
    with sqlite3.connect('example.db') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM employees')
        count = cursor.fetchone()[0]
        return count

employee = get_employee_by_id(1)
if employee:
    print(f"Found employee: {employee['name']}")

total_employees = safe_database_operation()
print(f"Total employees (safe method): {total_employees}")

# =============================================================================
# 15. CLEANUP
# =============================================================================

print("\n15. CLEANUP")

# Close connection
conn.close()
print("Database connection closed")

# =============================================================================
# SUMMARY OF KEY CONCEPTS
# =============================================================================

print("\n" + "="*50)
print("SQLITE3 QUICK REFERENCE:")
print("="*50)
print("""
CONNECTION:
  conn = sqlite3.connect('database.db')
  cursor = conn.cursor()

BASIC OPERATIONS:
  CREATE TABLE: CREATE TABLE name (columns...)
  INSERT: INSERT INTO table VALUES (?, ?, ?)
  SELECT: SELECT columns FROM table WHERE condition
  UPDATE: UPDATE table SET column = value WHERE condition
  DELETE: DELETE FROM table WHERE condition

IMPORTANT METHODS:
  cursor.execute(sql, params)     - Execute single query
  cursor.executemany(sql, data)   - Execute multiple queries
  cursor.fetchone()               - Get one result
  cursor.fetchall()               - Get all results
  cursor.fetchmany(n)             - Get n results
  conn.commit()                   - Save changes
  conn.rollback()                 - Undo changes
  conn.close()                    - Close connection

PANDAS INTEGRATION:
  pd.read_sql_query(sql, conn)    - Query to DataFrame
  df.to_sql(table, conn)          - DataFrame to table

SAFETY:
  - Always use ? placeholders for parameters
  - Handle exceptions with try/except
  - Use context managers when possible
  - Always commit() after changes
  - Close connections when done
""")

print("\nCrash course completed! Check 'example.db' file created in your directory.")