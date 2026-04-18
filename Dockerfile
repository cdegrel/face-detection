FROM python:3.9

WORKDIR /app

# Installer les dépendances système minimales
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de l'application
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exposer le port
EXPOSE 5005

# Lancer l'application
CMD ["python3", "app_flask.py"]
