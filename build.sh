#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Collect static files
cd BACKEND
python manage.py collectstatic --no-input

# Run migrations (if you were using a DB, but good practice to keep)
python manage.py migrate
