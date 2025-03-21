# Utiliser une image de base avec Python
FROM python:3.9-slim


# RUN apt-get update && apt-get install -y swig curl
RUN apt-get update && apt-get install -y swig gcc libpulse-dev

# Installer Ollama et s'assurer qu'il est disponible dans le PATH
RUN curl -fsSL https://ollama.com/install.sh | sh && \
    ln -sf /usr/local/bin/ollama /usr/bin/ollama

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt ./
COPY llmhackv1.py ./
COPY tools2.py ./
COPY README.md ./

# Installer les dépendances
RUN pip install --upgrade "pip<24.1"
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port de l’API FastAPI
EXPOSE 8000

# Définir la commande de lancement - exécuter ollama en arrière-plan puis votre application
CMD bash -c "ollama serve & sleep 5 && ollama pull llama2:latest && python llmhackv1.py"