FROM python:3.9-slim as build

RUN pip install virtualenv
RUN virtualenv /env

ENV PATH="/env/bin:$PATH"

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:3.9-slim
COPY --from=build /env /env
ENV PATH="/env/bin:$PATH"

WORKDIR /app
COPY . /app
COPY model /app/model

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]