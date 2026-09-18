# 🖥️ Smart Storage Manager (SSM)

Smart Storage Manager (SSM) is an intelligent, automated desktop infrastructure optimization tool built using **Supervised Machine Learning**. Instead of relying on rigid, unoptimized hardcoded rules (e.g., `if size > 100MB`), SSM analyzes complex correlations between multiple file system dimensions to help users declutter their machines and forecast future hardware constraints.

The project implements a **dual-core ML pipeline**:
1. **Classification Module:** A champion **Gradient Boosting Classifier (89% Accuracy)** that scans live directories recursively, handles file-level optimization, flags redundancies, calculates MD5 hashes for duplicate detection, tracks file access stagnation, and outputs granular analytical explanations in the terminal.
2. **Regression Module:** A continuous **Linear Regression Forecaster** that learns local storage consumption trajectory parameters month-over-month, calculates storage expansion velocity (+GB/month), and projects trend lines forward to estimate the exact time remaining (in months or years) before hard drive capacity exhaustion.

---

## Key Features

### 1. Machine Learning Driven Classification
* **89% Accurate Inference Engine:** Uses an optimized Gradient Boosting Classifier trained on programmatically engineered balanced datasets to evaluate unseen local files.
* **Granular Explainable AI (XAI):** Each flagged file is appended with an actionable textual reason explaining *why* the algorithm suggested a specific maintenance tier.
* **MD5 Hashing Duplicate Detector:** Incorporates `hashlib` to read binary file blocks and compute cryptographic MD5 checksums, identifying exact byte-level duplicate counts regardless of file renames.
* **Interactive Visualization:** Renders clean, real-time Matplotlib bar dashboards displaying file class distributions instantly.

### 2. Predictive Trend Regression
* **Live System Telemetry Integration:** Uses Python's built-in `shutil` library to safely poll actual operating system hardware partitions (e.g., `C:` drive) in real time.
* **Time-Series Horizon Forecasting:** Models historical gigabyte changes over time to predict disk state boundaries for the upcoming month.
* **Adaptive Unit Conversions:** Calculates the exact timeline remaining before disk expiration. If space remains clear for more than a year, it dynamically switches units from months to years (`years_left = months_left / 12.0`).
* **Continuous Visual Projections:** Generates a streamlined Matplotlib trend graph connecting past multi-month history directly to future prediction thresholds.

---

## System Architecture Diagram

```
                        SMART STORAGE MANAGER (SSM)
                                     │
                        ┌────────────┴────────────┐
                        ▼                         ▼
             [Classification Module]      [Regression Module]
                        │                         │
                 Live Folder Scan          shutil.disk_usage()
                        │                         │
               Extract File Metrics       Pull Historical Logs
             (Size, Age, MD5 Hashes)              │
                        │               Linear Regression Engine
           Gradient Boosting Model                │
                        │                 Calculate Slope (GB/mo)
            ┌───────────┼───────────┐             │
            ▼           ▼           ▼     Predict Next Month Space
          KEEP       REVIEW      CLEANUP          │
                        │                 Project Expiry (Months/Years)
                        ▼                         │
               • Explanatory Reasons              ▼
               • Visual Distribution     Continuous Trend Line Plot
```

---

## Dataset Properties & Feature Selection
The Classification pipeline operates on four domain-driven numerical feature dimensions, avoiding unoptimized system noise (like OS permission bits or hardware block sizes):

* `file_size_mb`: Continuous float tracking physical space footprint.
* `days_since_modified`: Days elapsed since the last structural file write.
* `days_since_accessed`: Days elapsed since the last read/open trigger (Data Stagnation Metric).
* `duplicate_count`: Integer calculated via MD5 binary block hashing tracking exact file redundancy copy overlays.

---

## Technical Stack
* **Core Language:** Python 3.10+
* **Data Wrangling:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`GradientBoostingClassifier`, `LinearRegression`)
* **File System Operations:** `os`, `time`, `shutil`, `hashlib`
* **Data Visualization:** Matplotlib

---

## Project Repository Structure
```directory
├── training_data.csv        
├── storage_history.csv      
├── SSM_classification.py    
├── SSM_regression.py        
├── README.md                
```

---

