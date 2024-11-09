PYTHON_VERSION := 3.11.2
ENV_NAME := movie_recommend

# Colors for terminal output
BLUE := \033[34m
GREEN := \033[32m
RED := \033[31m
RESET := \033[0m
YELLOW := \033[33m


# Path to the virtual environment’s pip and activate script
PIP := $(HOME)/.pyenv/versions/$(ENV_NAME)/bin/pip
ACTIVATE := $(HOME)/.pyenv/versions/$(ENV_NAME)/bin/activate

all: help

help:
	@echo "$(BLUE)Available commands:$(RESET)"
	@echo "$(GREEN)make create-env$(RESET)       - Create the virtual environment with pyenv"
	@echo "$(GREEN)make install-deps$(RESET)     - Install dependencies into the virtual environment (activate first)"
	@echo "$(GREEN)make setup$(RESET)            - Create environment, activate it, and install dependencies"

# Create the virtual environment and provide activation instructions
create-env:
	@echo "$(BLUE)Creating virtual environment with pyenv...$(RESET)"
	pyenv install -s $(PYTHON_VERSION) # Install Python version if not already installed
	pyenv virtualenv -f $(PYTHON_VERSION) $(ENV_NAME)
	@echo "$(GREEN)Virtual environment created.$(RESET)"
	@echo "$(BLUE)To activate the environment, run:$(RESET)"
	@echo "source $(ACTIVATE)"

# Install dependencies into the already created environment
install-deps:
	@echo "$(BLUE)Installing dependencies...$(RESET)"
	source $(ACTIVATE) && $(PIP) install --upgrade pip setuptools wheel
	source $(ACTIVATE) && $(PIP) install -r requirements/dev.txt
	@echo "$(GREEN)Dependencies installed successfully.$(RESET)"

activate-env:
	@echo "$(BLUE)==============================$(RESET)"
	@echo "$(GREEN)    To activate the environment, run:$(RESET)"
	@echo "$(YELLOW)      pyenv activate $(ENV_NAME)$(RESET)"
	@echo "$(BLUE)==============================$(RESET)"

# Combined command to create environment and install dependencies
setup: create-env install-deps activate-env
	@echo "$(GREEN)Environment setup and dependencies installed in one command.$(RESET)"
