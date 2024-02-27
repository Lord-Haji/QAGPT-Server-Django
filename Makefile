# Makefile for QAGPT-Backend-Django

# Setup poetry and install dependencies
install:
	@echo "Installing Project dependencies..."
	poetry install

# Database migrations
migrations:
	@echo "Making and applying database migrations..."
	poetry run python manage.py makemigrations
	poetry run python manage.py migrate

run:
	@echo "Starting Django server..."
	poetry run python manage.py runserver

# Run the Django exposed server at port 8000
run-exposed:
	@echo "Starting Django server..."
	poetry run python manage.py runserver 0.0.0.0:8000

# Formatting with Black
format:
	@echo "Formatting code with Black..."
	poetry run black .

# Linting with Flake8
lint:
	@echo "Linting code with Flake8..."
	poetry run pflake8

# Command to run all the necessary steps to start the application
start: install shell migrations run

.PHONY: install shell migrations run format lint start