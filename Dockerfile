FROM python:3.9
COPY requirements.txt /
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
ENV PORT 8000


CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}