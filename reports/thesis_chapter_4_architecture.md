# Chapter 4: System Architecture & Implementation

## 4.1 Architectural Overview

The OP-ECOM system is designed as a **real-time, low-latency inference service** decoupled from the core e-commerce platform. This Microservices-oriented architecture ensures that heavy machine learning computations do not degrade the performance of the main application (the "Shop").

The system follows a **Event-Driven / Request-Response hybrid pattern**:
1.  **Asynchronous Data Collection**: User behavior is captured non-blocking via `tracker.js`.
2.  **Synchronous Inference**: Purchase intent predictions are requested on-demand via a REST API.

![System Architecture Diagram]
*(Note: Imagine a diagram here showing User -> Store -> Tracker API -> MariaDB & Prediction API -> ONNX Model)*

---

## 4.2 Component Design

The system comprises four distinct layers, each capable of independent scaling:

### 4.2.1 Client Layer (The Data Source)
*   **Demo E-commerce Store (Vite)**: A lightweight Vanilla JS application that simulates a real shopping environment. It serves as the "Producer" of behavioral data.
*   **JavaScript Tracker (`tracker.js`)**: A generic, drop-in agent embedded in the client. It aggregates events (clicks, scrolls) and manages session state (Active/Idle) to reduce network noise.
*   **Prediction Dashboard (Next.js)**: A specialized administrative UI for identifying high-value sessions in real-time.

### 4.2.2 Service Layer (The Gateway)
We utilize **FastAPI** as the central gateway due to its asynchronous `uvloop` kernel, which offers performance comparable to NodeJS and Go.
*   **Tracker API (Port 8001)**: Handles high-throughput write operations (logging pageviews). It uses `SQLAlchemy` with connection pooling to manage MariaDB transactions efficiently.
*   **Prediction API (Port 8000)**: A dedicated read-heavy service for ML inference. It implements strict schema validation using **Pydantic** to prevent "garbage-in/garbage-out" errors before they reach the model.

### 4.2.3 Inference Engine (The Core)
To meet the strict <10ms latency requirement without GPUs, we bypassed standard PyTorch eager execution in favor of **ONNX Runtime (ORT)**.
*   **Model**: TabM (Tabular Model with Multiplicative Attention), pruned to an ensemble size of 4.
*   **Execution Provider**: `CPUExecutionProvider` tailored for AVX2 instruction sets.
*   **Threading Strategy**: We strictly set `intra_op_num_threads=1` and `inter_op_num_threads=1` to minimize context-switching overhead, relying instead on API-level parallelism (Uvicorn workers) for throughput.

### 4.2.4 Data Persistence Layer
*   **MariaDB (Port 3307)**: Chosen for its robust ACID compliance and row-level locking.
*   **Schema Design**: Normalized `sessions` and `page_views` tables ensure data integrity. An aggregated `sessions_export` view provides a snapshot compatible with the UCI dataset format for model retraining.

---

## 4.3 Data Pipeline & Flow

The lifecycle of a single prediction follows this critical path:

1.  **Ingestion**: User completes an action (e.g., views 'Product A' for 30s). `tracker.js` sends a generic payload.
2.  **Aggregation**: The Tracker API accepts the payload and updates the `sessions` row (incrementing `product_related_duration`).
3.  **Trigger**: When an inference is requested (e.g., on `checkout` load), the client calls `POST /predict`.
4.  **Preprocessing**: The Prediction API receives the raw session vector.
    *   *Normalization*: `StandardScaler` transforms numerical inputs.
    *   *Encoding*: `OneHotEncoder` transforms categorical inputs (Region, Month).
5.  **Inference**: The preprocessed vector (1x65 float32 tensor) is passed to ONNX Runtime. Evaluation completes in **~0.31ms**.
6.  **Action**: The probability (e.g., 0.85) is returned. The UI logic determines if an intervention (Discount Popup) is required.

---

## 4.4 Engineering Design Decisions

### 4.4.1 Why TabM over XGBoost?
While XGBoost offers competitive accuracy, **TabM (Neural Network)** allows for:
*   **Differentiable pipelines**: Future proofing for online training.
*   **Embedding layers**: Better handling of high-cardinality categorical features compared to One-Hot Encoding in trees.
*   **ONNX Compatability**: Deep learning models export more cleanly to standard inference runtimes than tree ensembles.

### 4.4.2 Why ONNX Runtime over PyTorch?
Our benchmarks revealed that the overhead of the Python Global Interpreter Lock (GIL) and PyTorch's dynamic graph construction accounted for **70% of total latency** in small-batch inference.
*   **PyTorch Latency**: ~1.38ms (Mean)
*   **ONNX Latency**: ~0.31ms (Mean)
*   **Speedup**: **4.4x improvement**, bringing the system comfortably below the real-time threshold.

---

## 4.5 Deployment Strategy

To ensure reproducibility across environments (Development vs. Production):
*   **Dockerization**: Each service (Backend, Tracker, Frontend, DB) runs in an isolated container defined in `docker-compose.yml`.
*   **Network Isolation**: A rigid internal network ensures only the API Gateway is exposed to the public; the Database is locked within the internal Docker network.

---

### 4.6 Summary of Specifications

| Component | Technology / Value |
| :--- | :--- |
| **Throughput** | ~2,400 Req/Sec (4 Workers) |
| **System Latency** | 1.61ms (p50) |
| **Model Size** | ~150 MB |
| **CPU Target** | Intel i7 (AVX2 supported) |
| **Container Base** | `python:3.9-slim` |

---
*This chapter demonstrates that the OP-ECOM project is not merely a data science experiment, but a fully realized, engineered software system capable of industrial deployment.*
