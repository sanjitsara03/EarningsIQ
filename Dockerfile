FROM python:3.12-slim AS base

# Stage 1: Build the React frontend
FROM node:20-slim AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python app with built frontend copied in
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

# Copy source and built frontend
COPY . .
COPY --from=frontend-builder /frontend/dist ./dist

EXPOSE 8000

CMD ["bash", "start.sh"]
