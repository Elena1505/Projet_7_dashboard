# Use an official Python runtime as the parent image
FROM python:3.8-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./ /app/

# Install required packages
# Note: You can combine packages in a requirements.txt for a more organized approach
RUN pip install --upgrade pip && \
    pip install streamlit

# If you have a requirements.txt, uncomment the line below
RUN pip install -r requirements.txt

# Expose port 8501 for Streamlit
EXPOSE 8501

# Define environment variable
# This is to fix an issue with Streamlit in Docker
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Run streamlit on container startup
CMD ["streamlit", "run", "dashboard.py"]