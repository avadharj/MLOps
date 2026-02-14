#!/bin/bash
# =============================================================
# Airflow Lab 1 (Modified) - Setup Script
# Author: Arjun Avadhani
# =============================================================
# This script sets up and runs the Airflow Lab 1 pipeline
# using Docker Compose.
#
# Prerequisites:
#   - Docker Desktop installed and running
#   - docker compose available
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# =============================================================

set -e

echo "=============================================="
echo "  Airflow Lab 1 (Modified) - Setup"
echo "=============================================="

# Step 1: Create required directories
echo ""
echo "[1/5] Creating required directories..."
mkdir -p ./logs ./plugins ./config ./working_data ./dags/model

# Step 2: Set AIRFLOW_UID (Linux/Mac)
echo "[2/5] Setting AIRFLOW_UID..."
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    echo "AIRFLOW_UID=$(id -u)" > .env
    echo "  Set AIRFLOW_UID=$(id -u)"
else
    echo "AIRFLOW_UID=50000" > .env
    echo "  Set AIRFLOW_UID=50000 (Windows default)"
fi

# Step 3: Check Docker is running
echo "[3/5] Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo "  ERROR: Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi
echo "  Docker is running."

# Step 4: Check available memory
echo "[4/5] Checking system resources..."
docker run --rm "debian:bullseye-slim" bash -c 'numfmt --to iec $(echo $(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE))))' 2>/dev/null || echo "  (Could not check memory)"

# Step 5: Initialize and run Airflow
echo "[5/5] Initializing Airflow database..."
echo "  This may take a few minutes on first run..."
echo ""
docker compose up airflow-init

echo ""
echo "=============================================="
echo "  Starting Airflow services..."
echo "=============================================="
echo ""
echo "  Once you see the health check log line, visit:"
echo "    URL:      http://localhost:8080"
echo "    Username: airflow2"
echo "    Password: airflow2"
echo ""
echo "  DAG name:   Airflow_Lab1_Modified"
echo ""
echo "  To stop:    docker compose down"
echo "  To cleanup: docker compose down --volumes --rmi all"
echo ""
echo "=============================================="

docker compose up