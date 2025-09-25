import logging

# Configure logging once at program start
# logging.basicConfig(
    # level=logging.INFO,  # default threshold
    # format="%(asctime)s [%(levelname)s] %(message)s",  # message format
# )

logging.basicConfig(
    filename="mobility.log",
    filemode="a",  # append mode
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Usage
logging.debug("This is debug (hidden if level=INFO).")
logging.info("Processing trip data…")
logging.warning("Missing geometry for person X.")
logging.error("Failed to write parquet file.")
logging.critical("System is out of memory!")
