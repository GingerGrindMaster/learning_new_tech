# python image
FROM python:3.9-slim

# set work directory, so that all commands files atd pod timhle radkem  are in one place that container can access, for easy access i guess
WORKDIR /app

# zkopíruje všechny soubory z aktuálního lokálního adresáře do pracovního adresáře uvnitř kontejneru
COPY . .

# install Flask into image
RUN pip install Flask

# příkaz, který říká, co má kontejner dělat, když se spustí, tedy zde říká Pythonu, aby spustil soubor app.py
CMD ["python", "app.py"]









