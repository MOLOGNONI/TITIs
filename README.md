<p align="center">
  <img src="https://raw.githubusercontent.com/MOLOGNONI/TITIs/main/assets/titis_banner.png" alt="TITIs Banner">
</p>

<h1 align="center">TITIs: Advanced Statistical Analysis Framework</h1>
<p align="center">
  <em>A toolkit for demonstrating and applying advanced multivariate statistical techniques in analytical chemistry.</em>
</p>

---

## 🔬 Overview

**TITIs (Tool for Inference of Total Intensity of Internal Standard)** is a Python-based framework designed to showcase a battery of advanced statistical analyses relevant to analytical chemistry, particularly in the context of chromatography and mass spectrometry.

This project provides a modular, script-driven workflow to analyze simulated experimental data, covering everything from data generation and dependency analysis to complex multivariate techniques like MANOVA, Canonical Correlation, and Linear Discriminant Analysis. It serves as both a powerful educational tool and a practical template for building robust analytical data processing pipelines.

## 🎯 Core Features

- **Modular Architecture:** The logic is cleanly separated into modules for data generation, statistical analysis, visualization, and insight generation.
- **Advanced Statistical Battery:** Includes implementations and examples for:
  - Correlation Matrix Analysis
  - Multivariate Analysis of Variance (MANOVA)
  - Canonical Correlation Analysis (CCA)
  - Linear Discriminant Analysis (LDA)
- **Data Simulation:** A sophisticated data generation engine that simulates experimental results from various extraction techniques (e.g., QuEChERS, SPE) and instrumental parameters.
- **Automated Insights:** A dedicated module to automatically interpret the results of the statistical tests and provide actionable recommendations.
- **Rich Visualizations:** Generates a suite of plots, including heatmaps, boxplots, biplots, and 3D response surfaces to help visualize complex data relationships.
- **Reproducible Workflow:** A simple, script-based entry point (`run_analysis.py`) ensures the entire analysis pipeline can be run consistently.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- The required packages are listed in `requirements.txt`.

### Installation & Execution

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MOLOGNONI/TITIs.git
    cd TITIs
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the complete analysis pipeline:**
    ```bash
    python scripts/run_analysis.py
    ```

    This script will execute the entire workflow: it generates the simulated data, performs all statistical analyses, creates the visualizations (which will be displayed on screen), and prints the final insights and recommendations to the console.

## 📂 Project Structure

```
.
├── assets/
│   └── titis_banner.png
├── notebooks/
│   └── TITIs.ipynb          # The original Jupyter Notebook with the analysis concepts.
├── scripts/
│   └── run_analysis.py      # Main script to execute the full analysis pipeline.
├── static/                  # Assets for the web analyzer
├── titis/
│   ├── __init__.py
│   ├── analysis.py          # Core statistical analysis functions (MANOVA, LDA, etc.).
│   ├── data.py              # Data generation and simulation.
│   ├── insights.py          # Automated interpretation of results.
│   └── plotting.py          # Advanced visualization functions.
├── iso21748-analyzer.html   # Web-based uncertainty calculator tool.
├── requirements.txt         # Project dependencies.
└── README.md
```

---

<p align="center">
  "The data has revealed its fundamental patterns."
</p>
