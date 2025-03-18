# Utiliser une image de base avec Python
FROM python:3.9-slim

# Installer Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh


#telecharger un modele
RUN ollama run llama2:latest


# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt ./
COPY llmhackv1.py ./
COPY tools2.py ./
COPY README.md ./

# Installer les dépendances
RUN apt-get update && apt-get install -y swig
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port de l’API FastAPI
EXPOSE 8000

# Définir la commande de lancement
CMD ["python", "llmhackv1.py"]
