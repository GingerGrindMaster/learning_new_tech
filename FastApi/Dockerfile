FROM python:3.11-slim-buster

WORKDIR /app

#zkopíruje requirements.txt z vašeho lokálního systému do kontejneru, přičemž zachová název souboru
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

#kopiruje se zbytek aplikace po instalaci závislostí z requirements.txt
COPY . .

CMD ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
