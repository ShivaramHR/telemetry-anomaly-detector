# Telemetry Anomaly Detector

A high-performance, real-time machine learning pipeline for detecting degradation in industrial engines. This project bridges the gap between Data Science and Systems Engineering by training an Autoencoder in Python, exporting it via ONNX, and executing lightning-fast inference using a native Rust UDP server.

---

## Architecture

The system is split into two primary domains:

1. Python Simulator (Model/Notebooks/): Handles EDA, Autoencoder training, ONNX model export, and simulates a 100Hz real-time sensor data stream.

2. Rust Inference Engine (Model/rust/): A low-latency daemon that ingests raw byte packets via UDP, shapes the tensors, and runs the ONNX model using the ort crate to calculate reconstruction loss (MSE).

``` mermaid
graph TD
    %% Styling
    classDef python fill:#3572A5,stroke:#fff,stroke-width:2px,color:#fff;
    classDef rust fill:#DEA584,stroke:#fff,stroke-width:2px,color:#000;
    classDef file fill:#444,stroke:#fff,stroke-width:1px,color:#fff;

    subgraph Python_Side [Python: Research & Simulation]
        A[AutoEncoder.ipynb] -->|1. Train & Export| B[tf_auto_encoder.onnx]
        C[simulator.py] -->|2. Stream 60-byte UDP Packets| D[Rust Network Layer]
    end

    subgraph Rust_Side [Rust: Production Inference Engine]
        B -->|3. Read into Memory| E[main.rs Orchestrator]
        D -->|4. Parse Bytes to f32| E
        E --> F[ndarray Tensor Shaping]
        F --> G[ort Crate: Native Model Execution]
        G --> H[MSE Loss Calculation]
        H -->|5. If Score > Threshold| I[Append to alerts.log]
    end

    %% Apply classes
    class A,C python;
    class E,F,G,H,I rust;
    class B,D file;
```
---

## Project Structure

```
telemetry-anomaly-detection/
├── Readme.md                          # Documentation & flowcharts
├── CMAPSSData/                        # Raw engine degradation dataset
│   ├── damage propogation modelin.pdf # NASA dataset background & methodology
│   └── readme.txt                     # Dataset instructions
└── Model/                             # Core Machine Learning & Inference Pipeline
    ├── autoencoder.keras              # Native Keras 3 saved model architecture and weights
    ├── model/                         # Exported deployment models
    │   ├── tf_saved_model/            # Intermediate SavedModel directory format
    │   └── tf_auto_encoder.onnx       # Compiled ONNX execution graph for Rust
    ├── Notebooks/                     # Python Research & Simulation Environment
    │   ├── AutoEncoder.ipynb          # Training pipeline and architecture
    │   ├── EDA.ipynb                  # Exploratory Data Analysis & Preprocessing
    │   ├── test_data_preparation.ipynb # Data split & cleaning logic
    │   └── simulator.py               # 100Hz UDP Mock packet injector script
    └── rust/                          # RUST SIDE: Native High-Speed Production
        └── src/
            └── main.rs                # Orchestrator, UDP Network layer, & ONNX Execution

```
---

## Phase 1: EDA & Preprocessing (Python)

To ensure the Autoencoder learns meaningful degradation patterns rather than static noise, the raw dataset (20631, 18) was heavily optimized:

Feature Engineering: Added RUL (Remaining Useful Life) and Degradation State.

Noise Reduction: Dropped sensor columns with constant variance that do not change over the engine's lifespan (sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19).

Operational Settings: Dropped op_setting columns as their variance was close to 0.

Final Target: The dataset was reduced to a lean (20631, 14) shape. RUL was dropped post-analysis to train the model purely on real-time features.

---

## Phase 2: The Autoencoder Model

The neural network utilizes a bottleneck architecture to learn the signature of a "healthy" engine:
14 → 16 → 8 → 4 → 8 → 14

The Challenge: Initial iterations yielded a flat, consistent loss curve (~0.0039) regardless of the engine's degradation state.

The Fix: Applied L2 Regularization. This forced the model to correctly reconstruct early-cycle (healthy) engines while failing to reconstruct late-cycle (failing) engines, causing a measurable spike in the MSE (Mean Squared Error).

---

## Phase 3: The High-Speed Server (Rust)

Why UDP instead of TCP?
In an industrial IoT environment, real-time data is strictly time-sensitive. If a packet is lost, TCP will halt the stream to recover 30ms-old data. UDP allows the system to drop the obsolete packet and instantly process the newest sensor readings.

The Payload Protocol:
The Python simulator fires a 60-byte payload per cycle:

Bytes 0-3: A 4-byte composite ID (unit_number * 1000 + time_in_cycles).

Bytes 4-59: 56 bytes representing the 14 f32 sensor readings.

Rust decodes these raw bytes instantly via chunks_exact(4) and f32::from_le_bytes, passing them to the ndarray and ort ONNX runtime for sub-millisecond inference.

---

## Phase 4: Dynamic Thresholding & Anomaly Detection

Finding the perfect static threshold proved challenging due to statistical distribution overlap:

Threshold 0.05: Too stubborn. Only flagged 3 out of 3,000 points. Missed early warning signs.

Threshold 0.03: Too sensitive. Flagged 231 points (7.7%). Captured too much normal operational variance.

The Early-Cycle Noise Problem:
Data analysis revealed that 63.9% of false alarms at the 0.03 threshold were occurring in the engine's early life cycle (<= 70%). Brand new engines often produce erratic sensor noise that triggers high MSE scores temporarily.

The Solution: Variable Thresholding (Predictive Maintenance)
To solve this, the pipeline is shifting to a dynamic threshold based on the time_in_cycles (extracted from the composite ID).

Early Cycles: The threshold remains loose to tolerate break-in noise.

Late Cycles: As the engine ages, the threshold dynamically tightens to catch micro-degradations right before failure.

--- 

## tech stack 
1. Python
2. Rust