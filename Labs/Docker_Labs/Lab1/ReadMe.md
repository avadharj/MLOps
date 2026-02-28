# Wine Classification with SVM (Dockerized)

## Overview

This project trains a **Support Vector Machine (SVM)** classifier on the **Wine dataset** from scikit-learn. The Wine dataset contains 178 samples of wine with 13 chemical features, classified into 3 categories. The entire training pipeline runs inside a Docker container.

### What Changed from the Original

| | Original | Updated |
|---|---|---|
| **Dataset** | Iris (150 samples, 4 features) | Wine (178 samples, 13 features) |
| **Model** | Random Forest | SVM (RBF kernel) |
| **Preprocessing** | None | StandardScaler (required for SVM) |
| **Output Files** | `iris_model.pkl` | `wine_model.pkl`, `wine_scaler.pkl` |

---

## Project Structure

```
project/
├── Dockerfile
├── README.md
└── src/
    ├── main.py
    └── requirements.txt
```

Place `main.py` and `requirements.txt` inside a `src/` folder. The `Dockerfile` goes in the project root.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running on your machine.

---

## Step-by-Step Instructions

### 1. Build the Docker Image

Open a terminal in the project root directory (where the `Dockerfile` is located) and run:

```bash
docker build -t lab1:v1 .
```

This builds the image and tags it as `lab1:v1`.

### 2. Run the Container

```bash
docker run lab1:v1
```

You should see output like:

```
Test Accuracy: 0.9722
The model training was successful
```

### 3. Save the Image to a Tar File

To export the Docker image as a portable `.tar` file:

```bash
docker save lab1:v1 > my_image.tar
```

### 4. Load the Image from a Tar File (on another machine)

```bash
docker load < my_image.tar
```

Then run it with `docker run lab1:v1`.

---

## Useful Docker Commands

| Command | Description |
|---|---|
| `docker images` | List all local images |
| `docker ps -a` | List all containers (running and stopped) |
| `docker rmi lab1:v1` | Remove the image |
| `docker system prune` | Clean up unused containers/images |