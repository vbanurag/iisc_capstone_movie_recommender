
# Setting Up Virtual Environments

## Using `venv` (Python's Built-In Virtual Environment)

1. Create a virtual environment:
   ```bash
   python3.11 -m venv venv
   ```

2. Activate the virtual environment:
   - On Unix/macOS:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

---

## Using `conda`

1. Create a conda environment from an `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the conda environment:
   ```bash
   conda activate movie-recommender
   ```

---

## Using `pyenv`

1. Install a specific version of Python with `pyenv`:
   ```bash
   pyenv install 3.11.2
   ```

2. Set the local Python version:
   ```bash
   pyenv local 3.11.2
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

---

## Alternative: Using a `make` Command

If `pyenv` is installed on your system, you can also set up your environment using a `make` command:

1. Run the following command to set up the environment:
   ```bash
   make setup
   ```

---
