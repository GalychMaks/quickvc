# QuickVC

## ⚙️ Installation

### 🔧 Set Up the Python Environment

1. **Clone the repository**

1. **Install `uv` — A fast Python package manager**\
   📖 [Installation guide](https://docs.astral.sh/uv/getting-started/installation/)

1. **Create and activate a virtual environment**

   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```

1. **Install dependencies**

- For inference only:

  ```bash
  uv pip install .
  ```

- For development and full tool support (includes `Taskfile` and `pre-commit`):

  ```bash
  uv pip install -e .[dev]
  ```

## 📚 References

- ⚡️ [QuickVC-VoiceConversion](https://github.com/quickvc/QuickVC-VoiceConversion) - Original repository
