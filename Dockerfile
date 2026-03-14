# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy only requirements.txt first (ensures Docker caching works better)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application
COPY . /app/


# Set working directory for Django
WORKDIR /app/BACKEND

# RUN python manage.py collectstatic --noinput  


# Expose the Django application port
EXPOSE 8000

# Run migrations, collect static files, and start the Django server
CMD ["sh", "-c", "python manage.py migrate && python manage.py collectstatic --noinput && python manage.py runserver 0.0.0.0:8000"]