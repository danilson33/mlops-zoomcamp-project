FROM prefecthq/prefect:2.11.5-python3.11

ENV PYTHONUNBUFFERED True

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .