# C2 Beacon Detection System

## 📌 Project Overview
The **C2 Beacon Detection System** is a modular cybersecurity tool designed to detect Command/Control (C2) beaconing activity in network traffic. It uses advanced statistical analysis (Fast Fourier Transform, Entropy, and Autocorrelation) to identify periodic communication patterns typical of malware callbacks.

The system consists of three main components:
1.  **Traffic Generation**: Scripts to simulate realistic beacon traffic and background noise.
2.  **Data Storage**: A PostgreSQL database (`c2db`) to store connection logs (`conn_log`).
3.  **Analysis & Dashboard**: A real-time analysis engine and a web-based dashboard for visualization.

## 🚀 Key Features
-   **Real-time Detection**: Analyzes network flows as they are logged.
-   **Multi-Factor Scoring**: Uses a "P-Score" (Periodicity Score) combining:
    -   **FFT Peak**: Strength of the dominant frequency.
    -   **Autocorrelation**: Repetitive patterns in time-series data.
    -   **Entropy**: Randomness of byte sizes and intervals.
-   **Interactive Dashboard**: Web interface to view alerts and active beacons.
-   **Simulation Tools**: Generate "perfect" or "realistic" beacon traffic for testing.

---

## 🛠️ Installation & Setup

### Prerequisites
-   **Linux OS** (Ubuntu/Debian recommended)
-   **Python 3.8+**
-   **PostgreSQL** database server

### 1. Clone & Initialize
```bash
# Clone the repository (if applicable)
git clone <repository_url>
cd c2-mini

# Run the setup script to create environments and directories
./setup_c2_project.sh
```

### 2. Configure Database
Ensure PostgreSQL is running and credentials match `config/database.conf`.
The default configuration expects:
-   **Database**: `c2db`
-   **User**: `c2user`
-   **Password**: `123`
-   **Host**: `localhost`

To fix database issues, run:
```bash
./fix_database_auth.sh
```

---

## 🖥️ Usage Guide

### 1. Generating Traffic (Simulation)
The project includes scripts to simulate network traffic for testing detection.

#### **Immediate Beacon Generation**
Generates 3 types of traffic (High/Med/Normal frequency) **immediately** into the database. Use this to see results instantly.
```bash
python3 generate_beacons_immediate.py
```

#### **Perfect Beacon Generation**
Generates mathematically perfect periodic beacons. Useful for validating the detection logic.
```bash
python3 generate_perfect_beacons.py
```

#### **Continuous Simulation**
Simulates a real beacon running over time.
```bash
python3 c2_beacon_generator.py
```

### 2. Running the Dashboard
The dashboard visualizes detection results and alerts.
```bash
python3 dashboard.py
```
-   **Access**: Open `http://localhost:5000` in your browser.
-   **Features**:
    -   **Analyze Now**: Triggers an immediate analysis of recent traffic.
    -   **System Status**: Shows active monitoring status.
    -   **Recent Alerts**: Lists detected beaconing IPs with their P-Scores.

### 3. Real-Time Analyzer
Runs in the background to continuously check for beacons.
```bash
python3 real_time_analyzer.py
```

---

## 📊 How It Works: The Detection Logic

The core analysis engine (`real_time_analyzer.py` / `beacon_analyzer.py`) transforms raw logs into a P-Score (Periodicity Score) ranging from 0 to 1.

### 1. Data Ingestion
-   Reads `conn_log` from PostgreSQL.
-   Filters for specific hosts and time windows (e.g., last 5 minutes).

### 2. Time Series Creation
-   Converts connection events into a time series (e.g., "bytes per second").

### 3. Statistical Analysis
-   **Fast Fourier Transform (FFT)**: Decomposes the signal into frequencies. A strong peak at a specific frequency (e.g., 0.1 Hz -> 10s interval) indicates periodicity.
-   **Autocorrelation**: Measures how well the signal correlates with itself at different time lags. High correlation at regular lags confirms a pattern.
-   **Entropy**: Measures randomness. Low entropy suggests automated machine behavior; high entropy suggests human behavior.

### 4. Scoring (P-Score)
The final score is a weighted sum:
```python
P_Score = (α * FFT_Peak) + (β * Autocorr_Max) + (γ * (1 - Entropy_Norm))
```
-   **Threshold**: A P-Score > **0.7** is typically flagged as a **BEACON**.

---

## 📂 Project Structure

```
c2-mini/
├── config/                  # Configuration files
│   └── database.conf        # DB connection settings
├── logs/                    # Application logs
├── output/                  # Analysis results (JSON)
├── templates/               # Web dashboard HTML
├── generate_beacons_immediate.py  # Quick traffic generator
├── generate_perfect_beacons.py    # Perfect traffic generator
├── dashboard.py             # Web dashboard entry point
├── real_time_analyzer.py    # Core analysis engine
├── beacon_analyzer.py       # Advanced offline analyzer
├── setup_c2_project.sh      # Setup script
└── README.md                # This file
```

## ❓ Troubleshooting

**Dashboard shows no data?**
1.  Run `generate_beacons_immediate.py` to ensure fresh data exists.
2.  Check `config/database.conf` for correct credentials.
3.  Ensure `dashboard.py` is running.

**Database connection failed?**
-   Check if PostgreSQL is active: `systemctl status postgresql`.
-   Run `./check_database_status.sh` to diagnose.