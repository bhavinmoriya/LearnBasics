UV is a new Python package manager written in Rust, maintained by Astral (the team behind Ruff), designed to replace tools like `pip`, `venv`, `pip tools`, and `pipx`.

Here are the notes listing the commands discussed in the video, along with their descriptions:

### UV Core Commands

| Command | Description |
| :--- | :--- |
| **`uv`** | Used to test if UV is installed and working; running it lists all available commands. |
| **`brew install UV`** | Used to install UV on a system using Homebrew (specific to Mac/Linux). |
| **`uv init [directory_name]`** | Initializes a new UV project, creating a project directory (e.g., `uv init new_app`). If no directory name is provided while inside an existing folder (`uv init`), it sets up the current directory as a new application. The default project type is `app`. |
| **`uv add [package_name]`** | Installs packages (e.g., `uv add flask request`). UV handles creating the virtual environment (`.venv` directory) automatically upon the first need. It also updates the `pyproject.toml` file to list dependencies and creates a `uv.lock` file. |
| **`uv add -r requirements.txt`** | Installs packages listed in a traditional `requirements.txt` file and migrates them into the `pyproject.toml` dependency list. |
| **`uv tree`** | Visualizes and helps understand dependencies by displaying a tree structure of main dependencies and their sub-dependencies. |
| **`uv run [script_name].py`** | Runs a Python script (e.g., `uv run main.py`). UV automatically finds and uses the project's virtual environment, even if it is not active. If the virtual environment has been deleted, this command quickly re-creates it, installs dependencies based on project/lock files, and then runs the code, all in one command. |
| **`uv sync`** | Recreates the exact environment using the `uv.lock` file. This is used when passing a project off to someone else so they can get their environment ready without running the application. |
| **`uv remove [package_name]`** | Removes a dependency (e.g., `uv remove flask`). This command updates the `pyproject.toml` and lock files. |

### UV Tool Commands (Replacing `pipx`)

| Command | Description |
| :--- | :--- |
| **`uv tool install [tool_name]`** | Installs a Python command-line tool (e.g., `uv tool install ruff`) in an isolated environment, making it available on the system path for use across multiple projects. |
| **`uv tool uninstall [tool_name]`** | Removes an installed tool (e.g., `uv tool uninstall ruff`). |
| **`uv tool run [tool_command]`** | Installs a tool in a *temporary* environment, runs the specific command (e.g., `ruff check`), and then cleans it up afterward, without permanently installing the tool. This is useful for testing or specific tasks. |
| **`uvx [tool_command]`** | A shortcut for the `uv tool run [tool_command]` command (e.g., `uvx ruff check`). |
| **`uv tool list`** | Shows the tools currently installed. |
| **`uv tool upgrade --all`** | Upgrades all installed tools at once. |

### UV Pip Subcommand (Transitional)

| Command | Description |
| :--- | :--- |
| **`uv pip install [package_name]`** | Acts as a direct, faster replacement for the standard `pip install` command. It is useful for transitioning gradually but does not offer UV's advanced functionality (like automatically creating `pyproject.toml` or lock files). |
| **`uv pip list`** | Lists installed packages, similar to the standard `pip list`. |
| **`uv pip freeze`** | Used to generate or overwrite the `requirements.txt` file when using the `uv pip` subcommand. |

### Traditional Python/Pip/Venv Commands (for comparison)

| Command | Description |
| :--- | :--- |
| **`python 3 -m venv venv`** | Creates a new virtual environment using the standard library tool `venv`. |
| **`source venv/bin/activate`** | Activates the newly created virtual environment (used on Mac/Linux). |
| **`pip install flask request`** | Installs packages using the traditional `pip` tool. |
| **`touch main.py`** | Creates a blank file (used to create a starter script). |
| **`pip freeze > requirements.txt`** | Creates a `requirements.txt` file by redirecting the output of `pip freeze`, which lists all installed dependencies (including sub-dependencies). |
| **`deactivate`** | Deactivates an active virtual environment. |
