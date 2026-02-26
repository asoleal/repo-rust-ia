# --- ETAPA 1: Compilación ---
# Debian Bookworm viene con Python 3.11 por defecto
FROM rust:1.84-slim as builder

# Instalar herramientas de Python del sistema
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Usar el Python 3.11 del sistema para compilar
RUN python3 -m venv .venv && \
    .venv/bin/pip install maturin && \
    .venv/bin/maturin build --release --out dist

# --- ETAPA 2: Ejecución ---
# Usamos exactamente la misma versión para que el Wheel sea compatible
FROM python:3.11-slim

WORKDIR /app

# Copiar e instalar el binario compilado (será cp311)
COPY --from=builder /app/dist/*.whl .
RUN pip install *.whl numpy && rm *.whl

# Copiar script de prueba
COPY prueba_persistencia.py .

CMD ["python", "prueba_persistencia.py"]
