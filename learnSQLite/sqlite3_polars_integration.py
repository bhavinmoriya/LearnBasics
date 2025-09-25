#!/usr/bin/env python3
"""
SQLite3 + Polars Integration Guide
Complete guide for using Polars with SQLite databases
"""

import sqlite3
import polars as pl
from datetime import datetime, date
import tempfile
import os

print("SQLite3 + Polars Integration Guide")
print("=" * 40)

# =============================================================================
# 1. SETUP - CREATE SAMPLE DATABASE
# =============================================================================

# Create temporary database for this example
db_path = "polars_example.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create sample tables
cursor.execute('''
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER,
        department TEXT,
        salary REAL,
        hire_date TEXT,
        is_active BOOLEAN
    )
''')

cursor.execute('''
    CREATE TABLE sales (
        sale_id INTEGER PRIMARY KEY,
        employee_id INTEGER,
        product TEXT,
        amount REAL,
        sale_date TEXT,
        FOREIGN KEY (employee_id) REFERENCES employees (id)
    )
''')

# Insert sample data
employees_data = [
    (1, 'Alice Johnson', 30, 'Engineering', 75000.0, '2023-01-15', True),
    (2, 'Bob Smith', 25, 'Marketing', 55000.0, '2023-03-20', True),
    (3, 'Charlie Brown', 35, 'Engineering', 85000.0, '2022-11-10', True),
    (4, 'Diana Wilson', 28, 'HR', 60000.0, '2023-02-05', False),
    (5, 'Eve Davis', 32, 'Engineering', 80000.0, '2022-08-15', True)
]

sales_data = [
    (1, 1, 'Software License', 1200.0, '2024-01-10'),
    (2, 2, 'Consulting', 2500.0, '2024-01-15'),
    (3, 1, 'Support Package', 800.0, '2024-01-20'),
    (4, 3, 'Software License', 1200.0, '2024-02-01'),
    (5, 5, 'Training', 1500.0, '2024-02-10'),
    (6, 2, 'Consulting', 3000.0, '2024-02-15')
]

cursor.executemany('INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?)', employees_data)
cursor.executemany('INSERT INTO sales VALUES (?, ?, ?, ?, ?)', sales_data)
conn.commit()

print("✓ Sample database created with employees and sales data")

# =============================================================================
# 2. READING FROM SQLITE TO POLARS
# =============================================================================

print("\n2. READING FROM SQLITE TO POLARS")
print("-" * 35)

# Method 1: Using pl.read_database() (Recommended)
print("2.1 Using pl.read_database():")
try:
    df_employees = pl.read_database(
        query="SELECT * FROM employees",
        connection=f"sqlite://{db_path}"
    )
    print("✓ Successfully read employees table:")
    print(df_employees)
except Exception as e:
    print(f"❌ Error with pl.read_database(): {e}")
    print("Note: pl.read_database() requires connectorx. Using alternative method...")
    
    # Method 2: Using sqlite3 + pl.from_records() (Fallback)
    print("\n2.2 Using sqlite3 + pl.from_records():")
    cursor.execute("SELECT * FROM employees")
    rows = cursor.fetchall()
    
    # Get column names
    cursor.execute("PRAGMA table_info(employees)")
    columns = [col[1] for col in cursor.fetchall()]
    
    # Create Polars DataFrame
    df_employees = pl.from_records(rows, schema=columns)
    print("✓ Successfully created Polars DataFrame:")
    print(df_employees)

# Method 3: Read with specific data types
print("\n2.3 Reading with explicit schema:")
cursor.execute("SELECT * FROM employees")
rows = cursor.fetchall()

# Define explicit schema for better type control
schema = {
    "id": pl.Int64,
    "name": pl.Utf8,
    "age": pl.Int32,
    "department": pl.Utf8,
    "salary": pl.Float64,
    "hire_date": pl.Utf8,  # We'll convert to Date later
    "is_active": pl.Boolean
}

df_employees_typed = pl.from_records(rows, schema=schema)

# Convert hire_date to proper Date type
df_employees_typed = df_employees_typed.with_columns(
    pl.col("hire_date").str.to_date("%Y-%m-%d").alias("hire_date")
)

print("✓ DataFrame with proper types:")
print(df_employees_typed.dtypes)
print(df_employees_typed)

# =============================================================================
# 3. COMPLEX QUERIES WITH POLARS POST-PROCESSING
# =============================================================================

print("\n3. COMPLEX QUERIES WITH POLARS")
print("-" * 32)

# Read sales data
cursor.execute("SELECT * FROM sales")
sales_rows = cursor.fetchall()
cursor.execute("PRAGMA table_info(sales)")
sales_columns = [col[1] for col in cursor.fetchall()]

df_sales = pl.from_records(sales_rows, schema=sales_columns)
df_sales = df_sales.with_columns(
    pl.col("sale_date").str.to_date("%Y-%m-%d").alias("sale_date")
)

print("3.1 Sales data loaded:")
print(df_sales)

# Join in SQLite vs Join in Polars comparison
print("\n3.2 JOIN Comparison:")

# SQLite JOIN
print("SQLite JOIN result:")
cursor.execute('''
    SELECT e.name, e.department, s.product, s.amount, s.sale_date
    FROM employees e
    JOIN sales s ON e.id = s.employee_id
    ORDER BY s.sale_date
''')
sqlite_join_rows = cursor.fetchall()
df_sqlite_join = pl.from_records(
    sqlite_join_rows, 
    schema=["name", "department", "product", "amount", "sale_date"]
)
df_sqlite_join = df_sqlite_join.with_columns(
    pl.col("sale_date").str.to_date("%Y-%m-%d")
)
print(df_sqlite_join)

# Polars JOIN
print("\nPolars JOIN result:")
df_polars_join = df_employees_typed.join(
    df_sales, 
    left_on="id", 
    right_on="employee_id"
).select([
    "name", "department", "product", "amount", "sale_date"
]).sort("sale_date")

print(df_polars_join)

# =============================================================================
# 4. POLARS ANALYTICS ON SQLITE DATA
# =============================================================================

print("\n4. POLARS ANALYTICS")
print("-" * 20)

# Complex aggregations that are easier in Polars
print("4.1 Department performance analysis:")
dept_analysis = df_polars_join.group_by("department").agg([
    pl.col("amount").sum().alias("total_sales"),
    pl.col("amount").mean().alias("avg_sale"),
    pl.col("amount").count().alias("sale_count"),
    pl.col("name").n_unique().alias("unique_sellers")
]).sort("total_sales", descending=True)

print(dept_analysis)

# Time series analysis
print("\n4.2 Monthly sales trend:")
monthly_sales = df_sales.with_columns(
    pl.col("sale_date").dt.strftime("%Y-%m").alias("month")
).group_by("month").agg([
    pl.col("amount").sum().alias("monthly_total"),
    pl.col("amount").count().alias("transaction_count")
]).sort("month")

print(monthly_sales)

# Advanced filtering and transformations
print("\n4.3 High-value transactions analysis:")
high_value_analysis = df_polars_join.filter(
    pl.col("amount") > 1000
).with_columns([
    pl.col("amount").rank(method="ordinal", descending=True).alias("amount_rank"),
    (pl.col("amount") / pl.col("amount").sum() * 100).round(2).alias("pct_of_total")
]).select([
    "name", "department", "product", "amount", "amount_rank", "pct_of_total"
])

print(high_value_analysis)

# =============================================================================
# 5. WRITING POLARS DATAFRAMES TO SQLITE
# =============================================================================

print("\n5. WRITING TO SQLITE")
print("-" * 21)

# Create a new DataFrame to insert
new_employees = pl.DataFrame({
    "id": [6, 7, 8],
    "name": ["Frank Miller", "Grace Lee", "Henry Kim"],
    "age": [29, 27, 31],
    "department": ["Finance", "Design", "Engineering"],
    "salary": [65000.0, 58000.0, 78000.0],
    "hire_date": [date(2023, 9, 1), date(2023, 10, 15), date(2023, 11, 20)],
    "is_active": [True, True, True]
})

print("5.1 New employees to insert:")
print(new_employees)

# Convert Polars DataFrame to records for SQLite insertion
print("\n5.2 Inserting into SQLite:")
records = new_employees.to_dicts()

for record in records:
    cursor.execute('''
        INSERT INTO employees (id, name, age, department, salary, hire_date, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        record["id"], record["name"], record["age"], 
        record["department"], record["salary"], 
        record["hire_date"].strftime("%Y-%m-%d"), record["is_active"]
    ))

conn.commit()
print(f"✓ Inserted {len(records)} new employees")

# Verify insertion
cursor.execute("SELECT COUNT(*) FROM employees")
total_employees = cursor.fetchone()[0]
print(f"Total employees in database: {total_employees}")

# =============================================================================
# 6. EFFICIENT BATCH OPERATIONS
# =============================================================================

print("\n6. BATCH OPERATIONS")
print("-" * 18)

# Create large dataset for batch operations
print("6.1 Creating large synthetic dataset:")
large_sales = pl.DataFrame({
    "sale_id": range(1000, 2000),
    "employee_id": [1, 2, 3, 4, 5] * 200,  # Cycle through employee IDs
    "product": ["Product A", "Product B", "Product C"] * 334,  # Mix of products
    "amount": pl.arange(1000, 2000, eager=True) * 0.75 + 100,  # Varied amounts
    "sale_date": pl.date_range(date(2024, 1, 1), date(2024, 12, 31), "3d", eager=True)[:1000]
})

print(f"Generated {len(large_sales)} sales records")
print("Sample data:")
print(large_sales.head())

# Batch insert using executemany
print("\n6.2 Batch inserting to SQLite:")
batch_records = [
    (row["sale_id"], row["employee_id"], row["product"], 
     row["amount"], row["sale_date"].strftime("%Y-%m-%d"))
    for row in large_sales.to_dicts()
]

cursor.executemany(
    'INSERT INTO sales (sale_id, employee_id, product, amount, sale_date) VALUES (?, ?, ?, ?, ?)',
    batch_records
)
conn.commit()
print(f"✓ Batch inserted {len(batch_records)} sales records")

# =============================================================================
# 7. PERFORMANCE COMPARISON
# =============================================================================

print("\n7. PERFORMANCE COMPARISON")
print("-" * 26)

import time

# SQLite aggregation
start_time = time.time()
cursor.execute('''
    SELECT department, AVG(salary) as avg_salary, COUNT(*) as count
    FROM employees 
    WHERE is_active = 1
    GROUP BY department
    ORDER BY avg_salary DESC
''')
sqlite_result = cursor.fetchall()
sqlite_time = time.time() - start_time

# Polars aggregation
start_time = time.time()
cursor.execute("SELECT * FROM employees")
all_employees = cursor.fetchall()
cursor.execute("PRAGMA table_info(employees)")
emp_columns = [col[1] for col in cursor.fetchall()]

df_all_employees = pl.from_records(all_employees, schema=emp_columns)
polars_result = df_all_employees.filter(
    pl.col("is_active") == 1
).group_by("department").agg([
    pl.col("salary").mean().alias("avg_salary"),
    pl.col("salary").count().alias("count")
]).sort("avg_salary", descending=True)
polars_time = time.time() - start_time

print(f"SQLite aggregation time: {sqlite_time:.4f} seconds")
print(f"Polars aggregation time: {polars_time:.4f} seconds")
print("\nSQLite result:")
for row in sqlite_result:
    print(f"  {row}")
print("\nPolars result:")
print(polars_result)

# =============================================================================
# 8. BEST PRACTICES FOR SQLITE + POLARS
# =============================================================================

print("\n8. BEST PRACTICES")
print("-" * 17)

def efficient_sqlite_to_polars(db_path: str, query: str, chunk_size: int = 10000):
    """
    Efficiently read large SQLite results into Polars DataFrame
    """
    conn = sqlite3.connect(db_path)
    
    # Get total count first
    count_query = f"SELECT COUNT(*) FROM ({query})"
    total_rows = conn.execute(count_query).fetchone()[0]
    
    if total_rows <= chunk_size:
        # Small dataset - read all at once
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return pl.from_records(rows, schema=columns)
    
    # Large dataset - read in chunks
    print(f"Reading {total_rows} rows in chunks of {chunk_size}")
    frames = []
    
    for offset in range(0, total_rows, chunk_size):
        chunk_query = f"{query} LIMIT {chunk_size} OFFSET {offset}"
        cursor = conn.cursor()
        cursor.execute(chunk_query)
        rows = cursor.fetchall()
        
        if not frames:  # First chunk - get column names
            columns = [desc[0] for desc in cursor.description]
        
        if rows:
            chunk_df = pl.from_records(rows, schema=columns)
            frames.append(chunk_df)
    
    conn.close()
    return pl.concat(frames) if frames else pl.DataFrame()

# Example usage
print("8.1 Efficient large dataset reading:")
large_df = efficient_sqlite_to_polars(db_path, "SELECT * FROM sales")
print(f"✓ Read {len(large_df)} sales records efficiently")
print("Sample of large dataset:")
print(large_df.head())

def polars_to_sqlite_optimized(df: pl.DataFrame, table_name: str, db_path: str, 
                              if_exists: str = "append", chunk_size: int = 1000):
    """
    Efficiently write Polars DataFrame to SQLite
    """
    conn = sqlite3.connect(db_path)
    
    # Convert to records in chunks for memory efficiency
    total_rows = len(df)
    
    for i in range(0, total_rows, chunk_size):
        chunk = df.slice(i, chunk_size)
        records = chunk.to_dicts()
        
        if i == 0 and if_exists == "replace":
            # Create table schema from first chunk
            columns = list(records[0].keys())
            placeholders = ", ".join(["?" for _ in columns])
            drop_sql = f"DROP TABLE IF EXISTS {table_name}"
            create_sql = f"CREATE TABLE {table_name} ({', '.join(columns)})"
            conn.execute(drop_sql)
            conn.execute(create_sql)
        
        # Insert chunk
        columns = list(records[0].keys())
        placeholders = ", ".join(["?" for _ in columns])
        insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        
        values = [[record[col] for col in columns] for record in records]
        conn.executemany(insert_sql, values)
    
    conn.commit()
    conn.close()
    print(f"✓ Wrote {total_rows} rows to {table_name} in chunks of {chunk_size}")

# =============================================================================
# 9. SUMMARY AND CLEANUP
# =============================================================================

print("\n9. SUMMARY")
print("-" * 10)

print("""
KEY TAKEAWAYS FOR SQLITE + POLARS:

✓ READING FROM SQLITE:
  - Use pl.read_database() if connectorx is available
  - Fallback: sqlite3 + pl.from_records()
  - Define explicit schemas for better type control
  - Convert string dates to proper Date types

✓ COMPLEX OPERATIONS:
  - Simple queries: Better in SQLite
  - Complex analytics: Better in Polars
  - Joins: Both work well, choose based on data size

✓ WRITING TO SQLITE:
  - Convert DataFrames to dicts/records
  - Use executemany() for batch operations
  - Handle data type conversions carefully

✓ PERFORMANCE:
  - SQLite: Better for simple aggregations on large datasets
  - Polars: Better for complex transformations and analytics
  - Use chunking for very large datasets

✓ BEST PRACTICES:
  - Read data in chunks for memory efficiency
  - Use proper data types in Polars schemas
  - Leverage both tools for their strengths
  - Always close database connections
""")

# Cleanup
conn.close()
print(f"\n✓ Database connection closed")
print(f"✓ Example database saved as: {db_path}")
print("✓ You can explore the database with any SQLite browser!")

# Optional: Clean up the example database
# os.remove(db_path)
# print("✓ Example database cleaned up")