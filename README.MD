# QAGPT-Backend-Django

## Overview

QAGPT-Backend-Django is a Django-based backend application, designed to provide robust and efficient backend services. This application is configured with PostgreSQL for database management and utilizes Python and Poetry for environment setup and package management.

## Prerequisites

Before setting up the project, ensure you have Python and `make` installed on your system. This application requires Python 3.x.

### Installing Make

- **Ubuntu/Debian:**
  ```bash
  sudo apt-get install make
  ```

- **macOS:**
  ```bash
  brew install make
  ```

- **Windows:**<br />
  Install `make` via [Chocolatey](https://chocolatey.org/install):
  ```powershell
  choco install make
  ```

  Install `make` via [Scoop](https://scoop.sh/):
  ```powershell
  scoop install make
  ```

### Set Environment Variables

Before installing the project, you need to set the following environment variables:

- `ASSEMBLYAI_API_KEY`: Your AssemblyAI API key.
- `GOOGLE_API_KEY`: Your GoogleGenAI API key.
- `OPENAI_API_KEY`: Your OpenAI API key.

## Installation

### Install Poetry

Poetry is used for managing dependencies in a more efficient way. Install it using the following commands:

**Note:** python - is the Python binary. Change python to whatever Python is called in your system.

- **Linux/Unix:**
  ```bash
  curl -sSL https://install.python-poetry.org | python -
  ```

- **Windows:**
  ```powershell
  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
  ```

### Install WeasyPrint Dependencies

WeasyPrint is used for PDF generation. Install its dependencies, GTK and Pango, as described in the [WeasyPrint documentation](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html).

### Install and Setup PostgreSQL

This application uses PostgreSQL as its database. Install PostgreSQL and configure the database credentials in `autoqa/settings.py`.

## Running the Application

Follow these steps to get the application running:

1. Install dependencies:
   ```bash
   make install
   ```

2. Activate the poetry shell:
   ```bash
   make shell
   ```

3. Setup database migrations:
   ```bash
   make migrations
   ```

4. Run the server:
   ```bash
   make run
   ```
   Optionally, you can run the server on port 8000:
   ```bash
   make run-exposed
   ```

## Accessing the Application

- Access the Django Admin Panel at `localhost:8000/admin`. If hosted, use `[hostname]:[port]/admin`.

- For API documentation, visit:
  - Swagger UI: `localhost/api/schema/docs` or `[hostname]:[port]/api/schema/docs`
  - ReDoc: `localhost/api/schema/redoc` or `[hostname]:[port]/api/schema/redoc`

## Code Formatting and Linting

- **Black** is used as the autoformatting solution.
  Run it using:
  ```bash
  make format
  ```

- **Flake8** is used for linting.
  Run it using:
  ```bash
  make lint
  ```

## TODO

- Implement and configure unit tests.

## Author

This project was developed by [Yudhajit Sinha](https://github.com/lord-haji). For more information or inquiries, please contact [yudhajitsinha@outlook.com](mailto:yudhajitsinha@outlook.com).