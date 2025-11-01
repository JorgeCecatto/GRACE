FROM python:3.13.9
WORKDIR /workspaces/src/
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .