Great question 🙂 — logging vs. `print` is something every developer eventually hits. Here’s a **crash course on Python logging**:

---

## 🔎 Why not `print`?

* `print()` is **always visible** — you can’t easily turn it off in production.
* No concept of severity: errors, warnings, debug info all look the same.
* Hard to redirect: if you want messages in a file, you must manually redirect `stdout`.
* No timestamps or context unless you add them yourself.

---

## ✅ Why use `logging`?

The `logging` module gives you:

* **Levels** (severity): `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
* **Configurable output**: file, console, or both.
* **Formatting**: automatic timestamps, function name, line number.
* **Granularity**: you can silence debug logs in production without touching code.

---

## ⚡ Quickstart Example

```python
import logging

# Configure logging once at program start
logging.basicConfig(
    level=logging.INFO,  # default threshold
    format="%(asctime)s [%(levelname)s] %(message)s",  # message format
)

# Usage
logging.debug("This is debug (hidden if level=INFO).")
logging.info("Processing trip data…")
logging.warning("Missing geometry for person X.")
logging.error("Failed to write parquet file.")
logging.critical("System is out of memory!")
```

### Output:

```
2025-09-25 17:32:10 [INFO] Processing trip data…
2025-09-25 17:32:10 [WARNING] Missing geometry for person X.
2025-09-25 17:32:10 [ERROR] Failed to write parquet file.
```

---

## 📂 Writing to a log file

```python
logging.basicConfig(
    filename="mobility.log",
    filemode="a",  # append mode
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
```

Now all logs go into `mobility.log`.

---

## 🎚️ Levels in practice

* `DEBUG`: internal details (variable values, steps). Use for development.
* `INFO`: high-level progress updates (e.g., "Processed 1000 trips").
* `WARNING`: something unexpected but not fatal (e.g., missing field).
* `ERROR`: serious issue, current operation failed.
* `CRITICAL`: program may not continue.

You control **what minimum level** shows with `level=...` in `basicConfig`.

---

## ⚙️ Advanced (when projects grow)

* **Multiple handlers**: console + file simultaneously.
* **Per-module loggers**: each Python file can have its own logger:

  ```python
  logger = logging.getLogger(__name__)
  logger.info("Message from this module")
  ```
* **Rotating logs**: auto-split files when too big (`logging.handlers.RotatingFileHandler`).

---

📌 **Rule of thumb**:

* Use `print()` only for **quick debugging or throwaway scripts**.
* Use `logging` in **any real script, library, or production system**.

---

👉 Do you want me to take your **CSV-to-Parquet conversion script** and show how I’d replace all `print()` calls with `logging` in a clean way?
