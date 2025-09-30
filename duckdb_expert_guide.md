# DuckDB Expert Guide: Complete Mastery Path

## Table of Contents
1. [What is DuckDB?](#what-is-duckdb)
2. [Installation & Setup](#installation--setup)
3. [Core Concepts](#core-concepts)
4. [Basic Operations](#basic-operations)
5. [Advanced Querying](#advanced-querying)
6. [Performance Optimization](#performance-optimization)
7. [Data Import/Export](#data-importexport)
8. [Python Integration](#python-integration)
9. [Advanced Features](#advanced-features)
10. [Real-World Use Cases](#real-world-use-cases)
11. [Best Practices](#best-practices)

---

## What is DuckDB?

DuckDB is an **in-process SQL OLAP database management system** designed for analytical workloads. Think of it as "SQLite for analytics."

**Key Characteristics:**
- **Columnar storage** - optimized for analytical queries
- **Vectorized execution** - processes data in batches for speed
- **In-process** - no separate server needed
- **ACID compliant** - reliable transactions
- **Zero dependencies** - single binary/library
- **Supports complex queries** - window functions, CTEs, recursive queries
- **Multi-platform** - Python, R, Java, Node.js, C++, CLI

**When to Use DuckDB:**
- Analyzing CSV, Parquet, JSON files without loading into a database
- Local data analysis and exploration
- ETL pipelines and data transformations
- Embedded analytics in applications
- Replacing pandas for larger-than-memory datasets
- Quick prototyping of analytical queries

---

## Installation & Setup

### Python Installation
```python
pip install duckdb

# Import
import duckdb

# Create in-memory database
con = duckdb.connect()

# Create persistent database
con = duckdb.connect('my_database.db')
```

### CLI Installation
```bash
# macOS
brew install duckdb

# Linux
wget https://github.com/duckdb/duckdb/releases/download/v1.0.0/duckdb_cli-linux-amd64.zip
unzip duckdb_cli-linux-amd64.zip

# Run
./duckdb my_database.db
```

---

## Core Concepts

### 1. In-Process Architecture
Unlike PostgreSQL or MySQL, DuckDB runs **inside your application process**. No network overhead, no separate server management.

### 2. Columnar Storage
Data is stored by column, not by row. This makes analytical queries (aggregations, filters) extremely fast.

```
Row-oriented (MySQL):     Columnar (DuckDB):
[1, 'Alice', 25]         [1, 2, 3, ...]
[2, 'Bob', 30]           ['Alice', 'Bob', 'Charlie', ...]
[3, 'Charlie', 35]       [25, 30, 35, ...]
```

### 3. Vectorized Execution
Processes thousands of values at once using CPU SIMD instructions, rather than one row at a time.

### 4. Query Execution Model
```
SQL Query → Parser → Binder → Optimizer → Physical Plan → Execution Engine → Results
```

---

## Basic Operations

### Creating Tables

```sql
-- Create table from scratch
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name VARCHAR,
    department VARCHAR,
    salary DECIMAL(10,2),
    hire_date DATE
);

-- Insert data
INSERT INTO employees VALUES
    (1, 'Alice', 'Engineering', 95000, '2020-01-15'),
    (2, 'Bob', 'Sales', 75000, '2019-06-20'),
    (3, 'Charlie', 'Engineering', 105000, '2018-03-10');

-- Create table from query
CREATE TABLE high_earners AS
SELECT * FROM employees WHERE salary > 90000;

-- Create temporary table
CREATE TEMP TABLE temp_results AS
SELECT department, AVG(salary) as avg_salary
FROM employees GROUP BY department;
```

### Basic Queries

```sql
-- Simple SELECT
SELECT * FROM employees;

-- Filtering
SELECT name, salary
FROM employees
WHERE department = 'Engineering' AND salary > 90000;

-- Sorting
SELECT * FROM employees
ORDER BY salary DESC, hire_date ASC;

-- Aggregations
SELECT 
    department,
    COUNT(*) as employee_count,
    AVG(salary) as avg_salary,
    MAX(salary) as max_salary,
    MIN(salary) as min_salary
FROM employees
GROUP BY department
HAVING AVG(salary) > 80000;

-- DISTINCT
SELECT DISTINCT department FROM employees;
```

### Joins

```sql
-- Sample tables
CREATE TABLE departments (dept_id INTEGER, dept_name VARCHAR, manager VARCHAR);
CREATE TABLE projects (project_id INTEGER, dept_id INTEGER, budget DECIMAL);

-- INNER JOIN
SELECT e.name, d.dept_name, e.salary
FROM employees e
INNER JOIN departments d ON e.department = d.dept_name;

-- LEFT JOIN
SELECT e.name, p.project_id, p.budget
FROM employees e
LEFT JOIN projects p ON e.department = p.dept_id;

-- Multiple joins
SELECT e.name, d.dept_name, p.budget
FROM employees e
INNER JOIN departments d ON e.department = d.dept_name
LEFT JOIN projects p ON d.dept_id = p.dept_id;

-- Self join
SELECT e1.name as employee, e2.name as colleague
FROM employees e1
INNER JOIN employees e2 ON e1.department = e2.department
WHERE e1.id < e2.id;
```

---

## Advanced Querying

### Window Functions

```sql
-- Row numbering
SELECT 
    name,
    department,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as salary_rank
FROM employees;

-- Running totals
SELECT 
    name,
    salary,
    SUM(salary) OVER (ORDER BY hire_date) as running_total
FROM employees;

-- Moving averages
SELECT 
    hire_date,
    salary,
    AVG(salary) OVER (
        ORDER BY hire_date 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) as moving_avg_3month
FROM employees;

-- Lag/Lead
SELECT 
    name,
    salary,
    LAG(salary, 1) OVER (ORDER BY hire_date) as previous_salary,
    LEAD(salary, 1) OVER (ORDER BY hire_date) as next_salary,
    salary - LAG(salary, 1) OVER (ORDER BY hire_date) as salary_change
FROM employees;

-- NTILE for percentiles
SELECT 
    name,
    salary,
    NTILE(4) OVER (ORDER BY salary) as salary_quartile
FROM employees;
```

### Common Table Expressions (CTEs)

```sql
-- Simple CTE
WITH high_earners AS (
    SELECT * FROM employees WHERE salary > 90000
)
SELECT department, COUNT(*) as count
FROM high_earners
GROUP BY department;

-- Multiple CTEs
WITH 
    dept_stats AS (
        SELECT department, AVG(salary) as avg_salary
        FROM employees
        GROUP BY department
    ),
    company_avg AS (
        SELECT AVG(salary) as company_avg FROM employees
    )
SELECT 
    d.department,
    d.avg_salary,
    c.company_avg,
    d.avg_salary - c.company_avg as difference
FROM dept_stats d
CROSS JOIN company_avg c;

-- Recursive CTE (organizational hierarchy)
CREATE TABLE org_chart (employee_id INTEGER, manager_id INTEGER, name VARCHAR);

WITH RECURSIVE hierarchy AS (
    -- Base case: top-level managers
    SELECT employee_id, manager_id, name, 1 as level
    FROM org_chart
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case: employees reporting to previous level
    SELECT o.employee_id, o.manager_id, o.name, h.level + 1
    FROM org_chart o
    INNER JOIN hierarchy h ON o.manager_id = h.employee_id
)
SELECT * FROM hierarchy ORDER BY level, name;
```

### Subqueries

```sql
-- Scalar subquery
SELECT name, salary,
    (SELECT AVG(salary) FROM employees) as company_avg
FROM employees;

-- Subquery in WHERE
SELECT * FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Subquery with IN
SELECT * FROM employees
WHERE department IN (
    SELECT dept_name FROM departments WHERE manager = 'Alice'
);

-- Correlated subquery
SELECT e1.name, e1.salary
FROM employees e1
WHERE e1.salary > (
    SELECT AVG(e2.salary)
    FROM employees e2
    WHERE e2.department = e1.department
);

-- EXISTS
SELECT * FROM departments d
WHERE EXISTS (
    SELECT 1 FROM employees e
    WHERE e.department = d.dept_name
);
```

### Advanced Aggregations

```sql
-- GROUPING SETS
SELECT 
    department,
    EXTRACT(YEAR FROM hire_date) as year,
    COUNT(*) as count
FROM employees
GROUP BY GROUPING SETS (
    (department),
    (year),
    (department, year),
    ()
);

-- ROLLUP (hierarchical aggregations)
SELECT 
    department,
    EXTRACT(YEAR FROM hire_date) as year,
    COUNT(*) as count,
    SUM(salary) as total_salary
FROM employees
GROUP BY ROLLUP(department, year);

-- CUBE (all combinations)
SELECT 
    department,
    EXTRACT(YEAR FROM hire_date) as year,
    COUNT(*) as count
FROM employees
GROUP BY CUBE(department, year);

-- FILTER clause
SELECT 
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE salary > 90000) as high_earners,
    COUNT(*) FILTER (WHERE department = 'Engineering') as engineers
FROM employees;
```

---

## Performance Optimization

### 1. Indexes

```sql
-- DuckDB automatically creates indexes on primary keys
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email VARCHAR UNIQUE,
    name VARCHAR
);

-- Manual index creation (less common in DuckDB)
-- DuckDB is optimized for full scans, indexes less critical than in OLTP databases
```

### 2. Query Optimization

```sql
-- Use EXPLAIN to understand query plans
EXPLAIN SELECT * FROM employees WHERE salary > 90000;

-- EXPLAIN ANALYZE for actual execution statistics
EXPLAIN ANALYZE SELECT * FROM employees WHERE salary > 90000;

-- Use appropriate data types
-- Good: INTEGER, DECIMAL, DATE
-- Avoid: VARCHAR for numeric data

-- Filter early in query
-- Good
SELECT * FROM (
    SELECT * FROM large_table WHERE date > '2024-01-01'
) WHERE department = 'Sales';

-- Better - filter pushed down automatically
SELECT * FROM large_table 
WHERE date > '2024-01-01' AND department = 'Sales';
```

### 3. Columnar Storage Optimization

```sql
-- DuckDB automatically optimizes column storage
-- Access only needed columns
-- Good
SELECT name, salary FROM employees;

-- Wasteful
SELECT * FROM employees; -- when you only need 2 columns
```

### 4. Parallel Execution

```python
import duckdb

# DuckDB uses multiple threads automatically
con = duckdb.connect()

# Control thread count
con.execute("SET threads TO 4")

# Query runs in parallel automatically
result = con.execute("""
    SELECT department, COUNT(*) 
    FROM large_table 
    GROUP BY department
""").fetchall()
```

### 5. Memory Management

```python
# Set memory limit
con.execute("SET memory_limit = '4GB'")

# Enable/disable out-of-core processing
con.execute("SET temp_directory = '/path/to/temp'")
```

---

## Data Import/Export

### Reading Files

```python
import duckdb

con = duckdb.connect()

# CSV
df = con.execute("SELECT * FROM 'data.csv'").df()

# With options
df = con.execute("""
    SELECT * FROM read_csv_auto('data.csv',
        delim=',',
        header=True,
        sample_size=100000
    )
""").df()

# Parquet
df = con.execute("SELECT * FROM 'data.parquet'").df()

# Multiple Parquet files (glob pattern)
df = con.execute("SELECT * FROM 'data/*.parquet'").df()

# JSON
df = con.execute("SELECT * FROM 'data.json'").df()

# JSON Lines
df = con.execute("SELECT * FROM read_json_auto('data.jsonl')").df()

# Excel
con.execute("INSTALL spatial")  # Need extension
df = con.execute("SELECT * FROM st_read('data.xlsx')").df()
```

### Writing Files

```python
# Export to CSV
con.execute("COPY (SELECT * FROM employees) TO 'output.csv' (HEADER, DELIMITER ',')")

# Export to Parquet
con.execute("COPY (SELECT * FROM employees) TO 'output.parquet' (FORMAT PARQUET)")

# Export to JSON
con.execute("COPY (SELECT * FROM employees) TO 'output.json'")

# Partitioned Parquet (for large datasets)
con.execute("""
    COPY (SELECT * FROM large_table) 
    TO 'output_dir' (FORMAT PARQUET, PARTITION_BY (department, year))
""")
```

### Direct File Querying (No Import!)

```python
# Query CSV directly - no loading required!
result = con.execute("""
    SELECT department, AVG(salary) as avg_sal
    FROM 'employees.csv'
    WHERE hire_date > '2020-01-01'
    GROUP BY department
""").fetchall()

# Join multiple files
result = con.execute("""
    SELECT e.name, d.dept_name
    FROM 'employees.csv' e
    JOIN 'departments.csv' d ON e.dept_id = d.id
""").fetchall()

# Query Parquet with pushdown predicates (super fast!)
result = con.execute("""
    SELECT * FROM 'huge_file.parquet'
    WHERE date BETWEEN '2024-01-01' AND '2024-12-31'
    AND region = 'US'
""").fetchall()
```

---

## Python Integration

### Basic Usage

```python
import duckdb
import pandas as pd

# Create connection
con = duckdb.connect()

# Execute query, get results
result = con.execute("SELECT 42 as answer").fetchall()
print(result)  # [(42,)]

# Get as DataFrame
df = con.execute("SELECT 42 as answer").df()

# Get as arrow table
arrow_table = con.execute("SELECT 42 as answer").arrow()

# Get as numpy array
numpy_array = con.execute("SELECT 42 as answer").fetchnumpy()
```

### Working with Pandas

```python
import pandas as pd
import duckdb

# Create sample DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [70000, 80000, 90000]
})

# Query DataFrame directly with DuckDB!
result = duckdb.query("""
    SELECT name, salary * 1.1 as new_salary
    FROM df
    WHERE age > 25
""").df()

print(result)

# Register DataFrame as view
con = duckdb.connect()
con.register('people', df)

result = con.execute("""
    SELECT AVG(salary) as avg_salary FROM people
""").fetchall()

# Multiple DataFrames
df1 = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'id': [1, 2], 'dept': ['Eng', 'Sales']})

result = duckdb.query("""
    SELECT df1.name, df2.dept
    FROM df1 JOIN df2 ON df1.id = df2.id
""").df()
```

### Replacement for Pandas Operations

```python
import duckdb
import pandas as pd

# Load large CSV (pandas would load into memory)
# DuckDB streams it!
large_df = duckdb.query("""
    SELECT * FROM 'large_file.csv'
    WHERE date > '2024-01-01'
""").df()

# Complex aggregations (faster than pandas for large data)
result = duckdb.query("""
    SELECT 
        department,
        EXTRACT(YEAR FROM date) as year,
        COUNT(*) as count,
        AVG(sales) as avg_sales
    FROM 'sales_data.csv'
    GROUP BY department, year
    ORDER BY year DESC, avg_sales DESC
""").df()

# Window functions (cleaner than pandas)
result = duckdb.query("""
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY sales DESC) as rank
    FROM sales_df
""").df()
```

### Performance Comparison

```python
import time
import pandas as pd
import duckdb

# Pandas approach
start = time.time()
df = pd.read_csv('large_file.csv')
result = df.groupby('category')['sales'].mean()
pandas_time = time.time() - start

# DuckDB approach
start = time.time()
result = duckdb.query("""
    SELECT category, AVG(sales) as avg_sales
    FROM 'large_file.csv'
    GROUP BY category
""").df()
duckdb_time = time.time() - start

print(f"Pandas: {pandas_time:.2f}s, DuckDB: {duckdb_time:.2f}s")
# DuckDB often 5-10x faster on large datasets
```

---

## Advanced Features

### 1. Full-Text Search

```python
con = duckdb.connect()

# Create FTS index
con.execute("""
    PRAGMA create_fts_index('documents', 'id', 'content')
""")

# Search
result = con.execute("""
    SELECT * FROM (
        SELECT *, fts_main_documents.match_bm25(id, 'machine learning') as score
        FROM documents
    ) WHERE score IS NOT NULL
    ORDER BY score DESC
""").df()
```

### 2. Time Series Functions

```python
# Generate time series
result = con.execute("""
    SELECT * FROM generate_series(
        TIMESTAMP '2024-01-01',
        TIMESTAMP '2024-12-31',
        INTERVAL '1 day'
    ) as t(date)
""").df()

# Time bucketing
result = con.execute("""
    SELECT 
        time_bucket(INTERVAL '1 hour', timestamp) as hour,
        AVG(value) as avg_value
    FROM sensor_data
    GROUP BY hour
    ORDER BY hour
""").df()

# Date arithmetic
result = con.execute("""
    SELECT 
        date,
        date + INTERVAL '7 days' as next_week,
        date - INTERVAL '1 month' as last_month
    FROM events
""").df()
```

### 3. JSON Operations

```python
# Query nested JSON
result = con.execute("""
    SELECT 
        json_extract(data, '$.name') as name,
        json_extract(data, '$.address.city') as city
    FROM json_table
""").df()

# JSON array operations
result = con.execute("""
    SELECT 
        json_extract(data, '$.tags[0]') as first_tag,
        json_array_length(data, '$.tags') as tag_count
    FROM json_table
""").df()
```

### 4. Regular Expressions

```python
# Pattern matching
result = con.execute("""
    SELECT * FROM emails
    WHERE email ~ '^[a-z]+@[a-z]+\.(com|org)$'
""").df()

# Extract with regex
result = con.execute("""
    SELECT 
        regexp_extract(phone, '\\d{3}-\\d{4}') as local_number
    FROM contacts
""").df()

# Replace with regex
result = con.execute("""
    SELECT 
        regexp_replace(text, '[0-9]+', 'XXX', 'g') as masked_text
    FROM documents
""").df()
```

### 5. Extensions

```python
import duckdb

con = duckdb.connect()

# Install extensions
con.execute("INSTALL httpfs")  # Read from HTTP/S3
con.execute("INSTALL spatial")  # GIS functions
con.execute("INSTALL json")     # JSON functions

# Load extensions
con.execute("LOAD httpfs")

# Read from S3
result = con.execute("""
    SELECT * FROM 's3://bucket/data.parquet'
""").df()

# Read from HTTP
result = con.execute("""
    SELECT * FROM 'https://example.com/data.csv'
""").df()
```

---

## Real-World Use Cases

### Use Case 1: Log Analysis

```python
import duckdb

con = duckdb.connect()

# Analyze web server logs directly from files
analysis = con.execute("""
    SELECT 
        DATE_TRUNC('hour', timestamp) as hour,
        status_code,
        COUNT(*) as request_count,
        AVG(response_time_ms) as avg_response_time,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time
    FROM 'logs/*.parquet'
    WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY hour, status_code
    ORDER BY hour DESC, request_count DESC
""").df()

print(analysis)
```

### Use Case 2: ETL Pipeline

```python
import duckdb

con = duckdb.connect()

# Extract, transform, load in one query
con.execute("""
    CREATE TABLE cleaned_sales AS
    SELECT 
        order_id,
        customer_id,
        product_id,
        CAST(amount AS DECIMAL(10,2)) as amount,
        DATE_TRUNC('day', order_date) as order_date,
        CASE 
            WHEN region IS NULL THEN 'Unknown'
            ELSE UPPER(TRIM(region))
        END as region
    FROM read_csv_auto('raw_sales/*.csv')
    WHERE amount > 0
        AND order_date >= '2024-01-01'
""")

# Export to Parquet partitioned by date
con.execute("""
    COPY cleaned_sales TO 'output/sales' 
    (FORMAT PARQUET, PARTITION_BY (order_date))
""")
```

### Use Case 3: Data Quality Checks

```python
import duckdb

con = duckdb.connect()

# Comprehensive data quality report
quality_report = con.execute("""
    SELECT 
        'Total Records' as check_name,
        COUNT(*) as value
    FROM 'data.csv'
    
    UNION ALL
    
    SELECT 
        'Null Customer IDs',
        COUNT(*) 
    FROM 'data.csv' 
    WHERE customer_id IS NULL
    
    UNION ALL
    
    SELECT 
        'Duplicate Records',
        COUNT(*) - COUNT(DISTINCT (order_id, customer_id))
    FROM 'data.csv'
    
    UNION ALL
    
    SELECT 
        'Invalid Dates',
        COUNT(*)
    FROM 'data.csv'
    WHERE order_date > CURRENT_DATE
    
    UNION ALL
    
    SELECT 
        'Negative Amounts',
        COUNT(*)
    FROM 'data.csv'
    WHERE amount < 0
""").df()

print(quality_report)
```

### Use Case 4: Time Series Analysis

```python
import duckdb

con = duckdb.connect()

# Stock price analysis with moving averages
result = con.execute("""
    SELECT 
        date,
        symbol,
        close_price,
        AVG(close_price) OVER (
            PARTITION BY symbol 
            ORDER BY date 
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) as sma_20,
        AVG(close_price) OVER (
            PARTITION BY symbol 
            ORDER BY date 
            ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
        ) as sma_50,
        close_price - LAG(close_price, 1) OVER (
            PARTITION BY symbol ORDER BY date
        ) as daily_change
    FROM 'stock_prices.parquet'
    WHERE symbol IN ('AAPL', 'GOOGL', 'MSFT')
    ORDER BY symbol, date
""").df()
```

---

## Best Practices

### 1. Connection Management

```python
# Use context manager for automatic cleanup
with duckdb.connect('my_db.db') as con:
    result = con.execute("SELECT * FROM table").df()
# Connection automatically closed

# For long-running applications, reuse connections
class DataAnalyzer:
    def __init__(self):
        self.con = duckdb.connect()
    
    def analyze(self, query):
        return self.con.execute(query).df()
    
    def close(self):
        self.con.close()
```

### 2. Query Parameterization

```python
# Use parameters to prevent SQL injection
department = "Engineering"
min_salary = 90000

# Correct - parameterized
result = con.execute("""
    SELECT * FROM employees 
    WHERE department = ? AND salary > ?
""", [department, min_salary]).df()

# Wrong - string concatenation (SQL injection risk!)
# result = con.execute(f"SELECT * FROM employees WHERE department = '{department}'").df()
```

### 3. File Format Choices

```python
# CSV: Human-readable, widely supported, but slow and large
# Use for: Small datasets, data exchange

# Parquet: Columnar, compressed, very fast
# Use for: Large datasets, analytical workloads (RECOMMENDED)

# JSON: Flexible schema, human-readable
# Use for: Semi-structured data, APIs

# Best practice: Convert CSV to Parquet
con.execute("""
    COPY (SELECT * FROM 'data.csv') 
    TO 'data.parquet' (FORMAT PARQUET, COMPRESSION 'ZSTD')
""")
```

### 4. Batch Processing

```python
# Process large datasets in chunks
chunk_size = 1000000

for offset in range(0, 10000000, chunk_size):
    chunk = con.execute(f"""
        SELECT * FROM large_table 
        LIMIT {chunk_size} OFFSET {offset}
    """).df()
    
    # Process chunk
    process_chunk(chunk)
```

### 5. Memory Considerations

```python
# Set memory limit for predictability
con.execute("SET memory_limit = '4GB'")

# Use temp directory for spilling to disk
con.execute("SET temp_directory = '/path/to/fast/disk'")

# Stream results instead of loading all into memory
for batch in con.execute("SELECT * FROM huge_table").fetch_arrow_reader(batch_size=10000):
    process_batch(batch)
```

### 6. Error Handling

```python
import duckdb

try:
    result = con.execute("SELECT * FROM nonexistent_table").df()
except duckdb.CatalogException as e:
    print(f"Table not found: {e}")
except duckdb.ParserException as e:
    print(f"SQL syntax error: {e}")
except duckdb.BinderException as e:
    print(f"Column/table binding error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Quick Reference: Common Patterns

### Pattern 1: Read-Process-Write

```python
import duckdb

con = duckdb.connect()

result = con.execute("""
    COPY (
        SELECT 
            customer_id,
            SUM(amount) as total_spent,
            COUNT(*) as order_count
        FROM 'orders.csv'
        WHERE order_date >= '2024-01-01'
        GROUP BY customer_id
    ) TO 'customer_summary.parquet' (FORMAT PARQUET)
""")
```

### Pattern 2: Multi-File Aggregation

```python
result = con.execute("""
    SELECT 
        DATE_TRUNC('month', date) as month,
        SUM(revenue) as monthly_revenue
    FROM 'sales/2024/*.parquet'
    GROUP BY month
    ORDER BY month
""").df()
```

### Pattern 3: Join External Files

```python
result = con.execute("""
    SELECT 
        c.customer_name,
        SUM(o.amount) as total_spent
    FROM 'orders.csv' o
    JOIN 'customers.parquet' c ON o.customer_id = c.id
    WHERE o.order_date >= '2024-01-01'
    GROUP BY c.customer_name
    ORDER BY total_spent DESC
    LIMIT 10
""").df()
```

### Pattern 4: Pivot Tables

```python
result = con.execute("""
    PIVOT (
        SELECT region, product, revenue
        FROM sales
    )
    ON product
    USING SUM(revenue)
    GROUP BY region
""").df()
```

---

## Performance Benchmarks (Approximate)

| Operation | Pandas | DuckDB | Speedup |
|-----------|--------|--------|---------|
| Read 1GB CSV | 15s | 3s | 5x |
| Group By (100M rows) | 45s | 5s | 9x |
| Join (10M x 10M) | 180s | 12s | 15x |
| Window Functions | 60s | 4s | 15x |

*Note: Actual performance depends on hardware, data, and query complexity*

---

## Learning Resources

1. **Official Documentation**: https://duckdb.org/docs/
2. **GitHub Examples**: https://github.com/duckdb/duckdb
3. **Practice**: Use DuckDB for your next data analysis project!

---

## Key Takeaways

✅ **DuckDB is perfect for:**
- Analyzing files (CSV, Parquet, JSON) without loading them
- Replacing pandas for large datasets
- ETL pipelines
- Embedded analytics
- Fast SQL queries on local data

❌ **DuckDB is NOT ideal for:**
- Transaction-heavy applications (use PostgreSQL)
- Real-time updates (OLTP workloads)
- Multi-user concurrent writes
- Applications requiring a database server

🎯 **Remember**: DuckDB's superpower is **querying files directly** with SQL at incredible speed!

---

**Practice Exercise**: Try this yourself!

```python
import duckdb

# Download sample data or use your own CSV
con = duckdb.connect()

# Query it directly - no loading!
result = con.execute("""
    SELECT 
        column1,
        COUNT(*) as count,
        AVG(column2) as average
    FROM 'your_file.csv'
    GROUP BY column1
    ORDER BY count DESC
    LIMIT 10
""").df()

print(result)
```

You're now a DuckDB expert! 🦆