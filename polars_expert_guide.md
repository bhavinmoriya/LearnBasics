# Complete Polars Expert Guide

## Table of Contents
1. [Fundamentals](#fundamentals)
2. [Data Types & Schema](#data-types--schema)
3. [DataFrame Operations](#dataframe-operations)
4. [Expression System](#expression-system)
5. [Lazy vs Eager Evaluation](#lazy-vs-eager-evaluation)
6. [Advanced Operations](#advanced-operations)
7. [Performance Optimization](#performance-optimization)
8. [Memory Management](#memory-management)
9. [Time Series & Temporal Data](#time-series--temporal-data)
10. [Working with Complex Data Types](#working-with-complex-data-types)
11. [I/O Operations](#io-operations)
12. [Advanced Patterns](#advanced-patterns)
13. [Testing & Debugging](#testing--debugging)
14. [Best Practices](#best-practices)

---

## Fundamentals

### Creating DataFrames

```python
import polars as pl
import numpy as np
from datetime import datetime, date

# From dictionary
df = pl.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "salary": [50000.0, 60000.0, 70000.0],
    "active": [True, False, True]
})

# From lists
df = pl.DataFrame([
    ["Alice", 25, 50000.0],
    ["Bob", 30, 60000.0],
    ["Charlie", 35, 70000.0]
], schema=["name", "age", "salary"])

# From numpy arrays
df = pl.DataFrame({
    "a": np.random.randn(100),
    "b": np.random.randint(0, 10, 100)
})

# From Series
s1 = pl.Series("names", ["Alice", "Bob", "Charlie"])
s2 = pl.Series("ages", [25, 30, 35])
df = pl.DataFrame([s1, s2])
```

### Basic Operations

```python
# Shape and info
print(df.shape)  # (rows, columns)
print(df.schema)  # Column names and types
print(df.dtypes)  # Data types
print(df.columns)  # Column names
print(df.describe())  # Summary statistics

# Head/tail
print(df.head(5))
print(df.tail(3))

# Sampling
print(df.sample(10))
print(df.sample(fraction=0.1))
```

---

## Data Types & Schema

### Core Data Types

```python
# Numeric types
pl.Int8, pl.Int16, pl.Int32, pl.Int64
pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64
pl.Float32, pl.Float64

# String and binary
pl.Utf8, pl.Binary

# Boolean
pl.Boolean

# Temporal types
pl.Date, pl.Time, pl.Datetime, pl.Duration

# Complex types
pl.List, pl.Struct, pl.Array, pl.Categorical
```

### Schema Definition and Casting

```python
# Define schema explicitly
schema = {
    "id": pl.UInt32,
    "name": pl.Utf8,
    "score": pl.Float64,
    "active": pl.Boolean,
    "created_at": pl.Datetime("ms")
}

df = pl.DataFrame(data, schema=schema)

# Casting
df = df.with_columns([
    pl.col("age").cast(pl.Float64),
    pl.col("salary").cast(pl.Int64),
    pl.col("name").cast(pl.Categorical)
])

# Automatic schema inference override
df = df.with_columns([
    pl.col("^.*_id$").cast(pl.UInt32),  # Regex pattern
    pl.col("date_*").str.strptime(pl.Date, "%Y-%m-%d")
])
```

---

## DataFrame Operations

### Selection

```python
# Select columns
df.select("name", "age")
df.select(pl.col("name"), pl.col("age"))
df.select(pl.col("^name|age$"))  # Regex

# Select by data type
df.select(pl.col(pl.Utf8))  # All string columns
df.select(pl.col(pl.NUMERIC_DTYPES))  # All numeric columns

# Exclude columns
df.select(pl.exclude("name"))
df.select(pl.exclude(["name", "age"]))

# Complex selections
df.select([
    pl.col("name").str.upper().alias("name_upper"),
    pl.col("age").cast(pl.Float64),
    pl.when(pl.col("salary") > 55000)
      .then(pl.lit("High"))
      .otherwise(pl.lit("Low"))
      .alias("salary_category")
])
```

### Filtering

```python
# Basic filtering
df.filter(pl.col("age") > 25)
df.filter(pl.col("name").str.contains("Alice"))

# Multiple conditions
df.filter(
    (pl.col("age") > 25) & (pl.col("salary") > 50000)
)

# String operations
df.filter(pl.col("name").str.starts_with("A"))
df.filter(pl.col("name").str.ends_with("e"))
df.filter(pl.col("name").str.len() > 5)

# Null handling
df.filter(pl.col("age").is_not_null())
df.filter(pl.col("name").is_in(["Alice", "Bob"]))

# Complex filtering
df.filter(
    pl.col("age").is_between(25, 35, closed="both")
)
```

### Sorting

```python
# Single column
df.sort("age")
df.sort("age", descending=True)

# Multiple columns
df.sort(["age", "salary"], descending=[True, False])

# By expression
df.sort(pl.col("name").str.len())
df.sort(pl.col("salary").rank(method="ordinal"))
```

### Grouping and Aggregation

```python
# Basic grouping
df.group_by("department").agg([
    pl.col("salary").mean().alias("avg_salary"),
    pl.col("age").max().alias("max_age"),
    pl.col("name").count().alias("count")
])

# Multiple grouping columns
df.group_by(["department", "level"]).agg([
    pl.col("salary").sum(),
    pl.col("bonus").first()
])

# Rolling operations
df.group_by("department").agg([
    pl.col("salary").rolling_mean(window_size=3).alias("rolling_avg")
])

# Custom aggregations
df.group_by("department").agg([
    pl.col("salary").map_elements(lambda x: x.quantile(0.75)).alias("q75"),
    pl.col("age").apply(lambda x: x.std()).alias("age_std")
])
```

---

## Expression System

### Core Expressions

```python
# Column reference
pl.col("name")
pl.col("^prefix_.*")  # Regex
pl.col(pl.Utf8)  # By type

# Literals
pl.lit(42)
pl.lit("hello")
pl.lit(None)

# Arithmetic
pl.col("a") + pl.col("b")
pl.col("salary") * 1.1
pl.col("age") ** 2

# Comparison
pl.col("age") > 25
pl.col("name") == "Alice"
pl.col("salary").is_between(40000, 80000)
```

### String Operations

```python
# String methods
pl.col("name").str.upper()
pl.col("name").str.lower()
pl.col("name").str.len()
pl.col("name").str.slice(0, 3)
pl.col("name").str.replace("old", "new")
pl.col("name").str.replace_all("pattern", "replacement")

# String splitting and joining
pl.col("full_name").str.split(" ")
pl.col("name_parts").list.join(", ")

# Pattern matching
pl.col("email").str.contains("@gmail.com")
pl.col("phone").str.extract(r"(\d{3})-(\d{3})-(\d{4})")
pl.col("text").str.extract_all(r"\d+")

# String parsing
pl.col("date_str").str.strptime(pl.Date, "%Y-%m-%d")
pl.col("number_str").str.parse_int()
```

### Conditional Logic

```python
# When-then-otherwise
pl.when(pl.col("age") < 18)
  .then(pl.lit("Minor"))
  .when(pl.col("age") < 65)
  .then(pl.lit("Adult"))
  .otherwise(pl.lit("Senior"))

# Multiple conditions
pl.when(
    (pl.col("age") > 25) & (pl.col("salary") > 50000)
).then(pl.lit("Qualified"))
.otherwise(pl.lit("Not Qualified"))

# Nested conditions
pl.when(pl.col("department") == "Engineering")
  .then(
      pl.when(pl.col("level") == "Senior")
      .then(pl.col("salary") * 1.2)
      .otherwise(pl.col("salary") * 1.1)
  )
  .otherwise(pl.col("salary"))
```

### Window Functions

```python
# Ranking
pl.col("salary").rank(method="ordinal").over("department")
pl.col("age").rank(method="dense", descending=True)

# Rolling operations
pl.col("sales").rolling_sum(window_size=7).over("store_id")
pl.col("temperature").rolling_mean(window_size=24)

# Cumulative operations
pl.col("sales").cumsum().over("date")
pl.col("count").cumcount()

# Lag and lead
pl.col("price").shift(1).over("symbol")  # Lag
pl.col("price").shift(-1).over("symbol")  # Lead

# First and last
pl.col("name").first().over("department")
pl.col("date").last().over("group")

# Percentiles over groups
pl.col("salary").quantile(0.75).over("department")
```

---

## Lazy vs Eager Evaluation

### Lazy Evaluation Benefits

```python
# Lazy DataFrame
lazy_df = pl.scan_csv("large_file.csv")

# Build query without execution
query = (
    lazy_df
    .filter(pl.col("age") > 25)
    .group_by("department")
    .agg([
        pl.col("salary").mean().alias("avg_salary"),
        pl.col("count").sum().alias("total_count")
    ])
    .sort("avg_salary", descending=True)
)

# Execute only when needed
result = query.collect()

# Streaming for large datasets
result = query.collect(streaming=True)
```

### Query Optimization

```python
# View query plan
print(query.explain())

# Optimized query plan
print(query.explain(optimized=True))

# Predicate pushdown example
lazy_df = pl.scan_parquet("*.parquet")
optimized_query = (
    lazy_df
    .filter(pl.col("date") >= "2023-01-01")  # Pushed down to file reading
    .select(["name", "amount", "date"])  # Only read needed columns
    .group_by("name")
    .agg(pl.col("amount").sum())
)
```

### Converting Between Lazy and Eager

```python
# Eager to lazy
df = pl.DataFrame({"a": [1, 2, 3]})
lazy_df = df.lazy()

# Lazy to eager
result = lazy_df.collect()

# Streaming collection
result = lazy_df.collect(streaming=True)

# Collect with specific options
result = lazy_df.collect(
    streaming=True,
    slice_pushdown=True,
    predicate_pushdown=True
)
```

---

## Advanced Operations

### Joins

```python
# Basic joins
df1.join(df2, on="id", how="inner")
df1.join(df2, on="id", how="left")
df1.join(df2, on="id", how="outer")
df1.join(df2, on="id", how="cross")

# Multiple keys
df1.join(df2, on=["id", "date"], how="inner")

# Different column names
df1.join(df2, left_on="user_id", right_on="id", how="left")

# Suffix handling
df1.join(df2, on="id", suffix="_right")

# Anti and semi joins
df1.join(df2, on="id", how="anti")  # Rows in df1 not in df2
df1.join(df2, on="id", how="semi")  # Rows in df1 that match df2

# Complex join conditions
df1.join(
    df2,
    on=(pl.col("id") == pl.col("user_id")) & 
       (pl.col("date") >= pl.col("start_date"))
)
```

### Concatenation

```python
# Vertical concatenation
pl.concat([df1, df2], how="vertical")
pl.concat([df1, df2], how="vertical_relaxed")  # Different schemas

# Horizontal concatenation
pl.concat([df1, df2], how="horizontal")

# Diagonal concatenation
pl.concat([df1, df2], how="diagonal")  # Fill missing columns with nulls

# With rechunk
pl.concat([df1, df2], rechunk=True)
```

### Pivoting and Melting

```python
# Pivot
df.pivot(
    index="date",
    columns="category",
    values="amount",
    aggregate_function="sum"
)

# Melt
df.melt(
    id_vars=["id", "date"],
    value_vars=["col1", "col2", "col3"],
    variable_name="metric",
    value_name="value"
)

# Unpivot (newer alternative to melt)
df.unpivot(
    index=["id", "date"],
    on=["col1", "col2", "col3"],
    variable_name="metric",
    value_name="value"
)
```

---

## Performance Optimization

### Memory-Efficient Operations

```python
# Use streaming for large datasets
lazy_df.collect(streaming=True)

# Chunked processing
def process_chunks(file_path, chunk_size=10000):
    reader = pl.read_csv_batched(file_path, batch_size=chunk_size)
    results = []
    for batch in reader:
        processed = batch.filter(pl.col("value") > 0)
        results.append(processed)
    return pl.concat(results)

# Memory-efficient aggregations
df.group_by("category").agg([
    pl.col("amount").sum(),
    pl.col("count").len()
]).collect(streaming=True)
```

### Optimization Techniques

```python
# Predicate pushdown
lazy_df = pl.scan_parquet("data.parquet")
result = (
    lazy_df
    .filter(pl.col("date") >= "2023-01-01")  # Applied during file reading
    .select(["col1", "col2"])  # Only read needed columns
    .collect()
)

# Use categoricals for repeated strings
df = df.with_columns(
    pl.col("category").cast(pl.Categorical)
)

# Efficient sorting
df.sort("date", maintain_order=True)  # Stable sort when needed

# Use lazy evaluation for complex pipelines
result = (
    pl.scan_csv("input.csv")
    .filter(pl.col("active") == True)
    .group_by("department")
    .agg(pl.col("salary").mean())
    .sort("salary", descending=True)
    .collect()
)
```

### Parallel Processing

```python
# Configure number of threads
pl.Config.set_thread_pool_size(8)

# Use concurrent operations
import concurrent.futures

def process_file(file_path):
    return pl.scan_csv(file_path).collect()

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_file, f) for f in file_paths]
    results = [f.result() for f in futures]
    combined = pl.concat(results)
```

---

## Memory Management

### Memory Monitoring

```python
# Check memory usage
print(df.estimated_size("mb"))
print(df.estimated_size("gb"))

# Memory-efficient data types
df = df.with_columns([
    pl.col("small_int").cast(pl.Int16),  # Instead of Int64
    pl.col("category").cast(pl.Categorical),  # Instead of Utf8
    pl.col("flag").cast(pl.Boolean)  # Instead of Int8
])

# Rechunk for memory efficiency
df = df.rechunk()
```

### Garbage Collection and Cleanup

```python
import gc

# Explicit cleanup
del df
gc.collect()

# Context manager for temporary operations
def process_large_data(file_path):
    with pl.Config(streaming_chunk_size=1000):
        return pl.scan_csv(file_path).collect(streaming=True)

# Memory-mapped files for very large data
lazy_df = pl.scan_csv("huge_file.csv", memory_map=True)
```

---

## Time Series & Temporal Data

### Date/Time Operations

```python
# Parse dates
df = df.with_columns([
    pl.col("date_str").str.strptime(pl.Date, "%Y-%m-%d"),
    pl.col("datetime_str").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
])

# Date components
df = df.with_columns([
    pl.col("date").dt.year().alias("year"),
    pl.col("date").dt.month().alias("month"),
    pl.col("date").dt.day().alias("day"),
    pl.col("date").dt.weekday().alias("weekday"),
    pl.col("date").dt.quarter().alias("quarter")
])

# Date arithmetic
df = df.with_columns([
    pl.col("date") + pl.duration(days=30),
    pl.col("date") - pl.duration(weeks=2),
    pl.col("end_date") - pl.col("start_date").alias("duration")
])
```

### Time Series Analysis

```python
# Resampling
df.group_by_dynamic(
    "date",
    every="1w",  # Weekly
    period="1w",
    offset="1d"
).agg([
    pl.col("value").mean().alias("avg_value"),
    pl.col("count").sum().alias("total_count")
])

# Rolling windows
df.with_columns([
    pl.col("price").rolling_mean(window_size=7).alias("ma_7"),
    pl.col("price").rolling_std(window_size=7).alias("std_7"),
    pl.col("volume").rolling_sum(window_size=7).alias("vol_7")
])

# Time-based filtering
df.filter(
    pl.col("date").is_between(
        datetime(2023, 1, 1),
        datetime(2023, 12, 31)
    )
)
```

### Business Calendar Operations

```python
# Business days
df = df.with_columns([
    pl.col("date").dt.is_business_day().alias("is_business_day"),
    pl.col("date").dt.add_business_days(5).alias("plus_5_bdays")
])

# Custom business calendar
from datetime import date
holidays = [date(2023, 1, 1), date(2023, 12, 25)]
df = df.with_columns([
    pl.col("date").dt.is_business_day(holidays=holidays)
])
```

---

## Working with Complex Data Types

### Lists and Arrays

```python
# Create list columns
df = pl.DataFrame({
    "id": [1, 2, 3],
    "scores": [[85, 90, 78], [92, 88, 95], [76, 82, 89]]
})

# List operations
df = df.with_columns([
    pl.col("scores").list.len().alias("num_scores"),
    pl.col("scores").list.mean().alias("avg_score"),
    pl.col("scores").list.max().alias("max_score"),
    pl.col("scores").list.get(0).alias("first_score")
])

# List transformations
df = df.with_columns([
    pl.col("scores").list.eval(
        pl.element().map_elements(lambda x: x * 1.1)
    ).alias("adjusted_scores")
])

# Explode lists to rows
df.explode("scores")

# Array operations (fixed-size lists)
df = df.with_columns([
    pl.col("scores").cast(pl.Array(pl.Float64, 3))
])
```

### Structs

```python
# Create struct columns
df = pl.DataFrame({
    "person": [
        {"name": "Alice", "age": 25, "city": "NYC"},
        {"name": "Bob", "age": 30, "city": "LA"},
        {"name": "Charlie", "age": 35, "city": "Chicago"}
    ]
})

# Struct operations
df = df.with_columns([
    pl.col("person").struct.field("name").alias("name"),
    pl.col("person").struct.field("age").alias("age")
])

# Nested struct operations
df = df.with_columns([
    pl.struct([
        pl.col("name"),
        pl.col("age") + 1
    ]).alias("updated_person")
])

# Unnest structs
df.unnest("person")
```

### Categorical Data

```python
# Convert to categorical
df = df.with_columns([
    pl.col("category").cast(pl.Categorical)
])

# Categorical operations
df = df.with_columns([
    pl.col("category").cat.set_ordering("lexical"),
    pl.col("category").cat.to_local()
])

# Global string cache for categoricals
with pl.StringCache():
    df1 = df1.with_columns(pl.col("cat").cast(pl.Categorical))
    df2 = df2.with_columns(pl.col("cat").cast(pl.Categorical))
    # Now categoricals are compatible across DataFrames
```

---

## I/O Operations

### Reading Data

```python
# CSV
df = pl.read_csv("file.csv")
df = pl.read_csv("file.csv", separator="|", quote_char="'")
df = pl.read_csv("file.csv", schema={"col1": pl.Int64, "col2": pl.Utf8})

# Parquet
df = pl.read_parquet("file.parquet")
df = pl.read_parquet("*.parquet")  # Multiple files
df = pl.read_parquet("file.parquet", columns=["col1", "col2"])

# JSON
df = pl.read_json("file.json")
df = pl.read_ndjson("file.ndjson")  # Newline-delimited JSON

# Excel
df = pl.read_excel("file.xlsx", sheet_name="Sheet1")

# Database
df = pl.read_database("SELECT * FROM table", connection_string)

# Lazy reading (recommended for large files)
lazy_df = pl.scan_csv("large_file.csv")
lazy_df = pl.scan_parquet("*.parquet")
```

### Writing Data

```python
# CSV
df.write_csv("output.csv")
df.write_csv("output.csv", separator="|", quote_char="'")

# Parquet
df.write_parquet("output.parquet")
df.write_parquet("output.parquet", compression="snappy")

# JSON
df.write_json("output.json")
df.write_ndjson("output.ndjson")

# Excel
df.write_excel("output.xlsx", worksheet="Sheet1")

# Database
df.write_database("table_name", connection_string)

# Streaming writes for large data
lazy_df.sink_csv("output.csv")
lazy_df.sink_parquet("output.parquet")
```

### Advanced I/O

```python
# Chunked reading
reader = pl.read_csv_batched("large_file.csv", batch_size=10000)
for batch in reader:
    # Process each batch
    processed = batch.filter(pl.col("value") > 0)
    processed.write_csv("processed_batch.csv", append=True)

# Cloud storage
df = pl.read_parquet("s3://bucket/file.parquet")
df = pl.read_csv("gs://bucket/file.csv")

# Compressed files
df = pl.read_csv("file.csv.gz")
df = pl.read_parquet("file.parquet.bz2")
```

---

## Advanced Patterns

### Custom Functions and UDFs

```python
# Simple UDF
def categorize_age(age):
    if age < 18:
        return "Minor"
    elif age < 65:
        return "Adult"
    else:
        return "Senior"

df = df.with_columns([
    pl.col("age").map_elements(categorize_age, return_dtype=pl.Utf8).alias("category")
])

# Vectorized operations (preferred)
df = df.with_columns([
    pl.when(pl.col("age") < 18)
    .then(pl.lit("Minor"))
    .when(pl.col("age") < 65)
    .then(pl.lit("Adult"))
    .otherwise(pl.lit("Senior"))
    .alias("category")
])

# Complex UDF with multiple inputs
def calculate_score(name, age, salary):
    base_score = len(name) * 10
    age_bonus = max(0, age - 25) * 2
    salary_bonus = salary / 10000
    return base_score + age_bonus + salary_bonus

df = df.with_columns([
    pl.struct(["name", "age", "salary"])
    .map_elements(
        lambda x: calculate_score(x["name"], x["age"], x["salary"]),
        return_dtype=pl.Float64
    )
    .alias("score")
])
```

### Advanced Aggregations

```python
# Multiple aggregation functions
df.group_by("category").agg([
    pl.col("value").sum().alias("total"),
    pl.col("value").mean().alias("average"),
    pl.col("value").std().alias("std_dev"),
    pl.col("value").quantile(0.5).alias("median"),
    pl.col("value").count().alias("count"),
    pl.col("value").n_unique().alias("unique_count")
])

# Custom aggregation
df.group_by("category").agg([
    pl.col("value").map_elements(
        lambda x: x.quantile(0.95) - x.quantile(0.05),
        return_dtype=pl.Float64
    ).alias("iqr_range")
])

# Conditional aggregation
df.group_by("category").agg([
    pl.col("value").filter(pl.col("active") == True).sum().alias("active_sum"),
    pl.col("value").filter(pl.col("active") == False).sum().alias("inactive_sum")
])
```

### Performance Patterns

```python
# Efficient filtering patterns
# Good: Use multiple filters
df.filter(pl.col("age") > 25).filter(pl.col("salary") > 50000)

# Better: Combine with &
df.filter((pl.col("age") > 25) & (pl.col("salary") > 50000))

# Efficient column selection
# Good: Select needed columns early
df.select(["col1", "col2", "col3"]).filter(pl.col("col1") > 0)

# String operations optimization
# Use categorical for repeated string operations
df = df.with_columns(pl.col("category").cast(pl.Categorical))

# Efficient joins
# Sort both DataFrames before joining large datasets
df1 = df1.sort("join_key")
df2 = df2.sort("join_key")
result = df1.join(df2, on="join_key")
```

---

## Testing & Debugging

### Testing DataFrames

```python
import pytest

def test_dataframe_operations():
    # Create test data
    df = pl.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "salary": [50000, 60000, 70000]
    })
    
    # Test filtering
    result = df.filter(pl.col("age") > 25)
    assert result.height == 2
    assert result["name"].to_list() == ["Bob", "Charlie"]
    
    # Test aggregation
    avg_salary = df.select(pl.col("salary").mean()).item()
    assert avg_salary == 60000

def test_schema():
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["A", "B", "C"]
    })
    
    expected_schema = {"id": pl.Int64, "name": pl.Utf8}
    assert df.schema == expected_schema

# Property-based testing with hypothesis
from hypothesis import given, strategies as st

@given(
    ages=st.lists(st.integers(min_value=18, max_value=100), min_size=1, max_size=100)
)
def test_age_categorization(ages):
    df = pl.DataFrame({"age": ages})
    result = df.with_columns([
        pl.when(pl.col("age") < 30)
        .then(pl.lit("Young"))
        .otherwise(pl.lit("Old"))
        .alias("category")
    ])
    
    assert result.height == len(ages)
    assert set(result["category"].to_list()) <= {"Young", "Old"}
```

### Debugging Techniques

```python
# Debug query plans
query = df.lazy().filter(pl.col("age") > 25).group_by("department").agg(pl.col("salary").mean())
print(query.explain())  # See query plan
print(query.explain(optimized=True))  # See optimized plan

# Debug data types
print(df.dtypes)
print(df.schema)

# Debug memory usage
print(f"Memory usage: {df.estimated_size('mb')} MB")

# Debug with intermediate steps
df_step1 = df.filter(pl.col("age") > 25)
print(f"After filter: {df_step1.height} rows")

df_step2 = df_step1.group_by("department").agg(pl.col("salary").mean())
print(f"After groupby: {df_step2.height} rows")

# Debug with assertions
assert df.height > 0, "DataFrame is empty"
assert "required_column" in df.columns, "Missing required column"
assert df.filter(pl.col("age").is_null()).height == 0, "Found null ages"
```

---

## Best Practices

### Code Organization

```python
# Use constants for column names
class Columns:
    ID = "id"
    NAME = "name"
    AGE = "age"
    SALARY = "salary"

# Function for complex operations
def clean_employee_data(df: pl.DataFrame) -> pl.DataFrame:
    return (
```
## Date And Time
```python
import polars as pl

# Sample data
df = pl.DataFrame({
    "a": ["09:30:00", "14:15:30", "18:45:00"],  # time strings
    "b": ["2024-01-15", "2024-01-16", "2024-01-17"]  # date strings
})

# Method 1: Concatenate strings then parse
df_method1 = df.with_columns(
    pl.concat_str([
        pl.col("b"), 
        pl.lit(" "), 
        pl.col("a")
    ]).str.to_datetime().alias("c")
)
```

        