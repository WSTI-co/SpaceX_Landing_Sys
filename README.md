# SpaceX Landing System

This repository contains the **SpaceX Landing System**, a Python-based program designed to simulate and monitor SpaceX booster landings under varying wind conditions. The program performs precise calculations, records flight data, and provides real-time visualizations to ensure efficient booster landings. This project is a prototype and open for community contributions.

## Features:
1. **Real-Time Data Visualization:**
   - Displays telemetry data in a live **Matplotlib** graph.
   - Visualizes fuel consumption (time vs. fuel) and flight path (altitude vs. time) in real time.

2. **Data Logging:**
   - Records flight data every 0.05 seconds and saves it in an **Excel file** for detailed analysis.

3. **Abort Safety Mechanism:**
   - Automatically aborts landing when wind conditions exceed a critical threshold, requiring a program restart.

4. **Touchdown Registration:**
   - Confirms a successful landing with a "Touchdown" output.

## Requirements:
- **Python 3.8 or higher**
- Installed libraries:
  - `os` and `sys`: For system and file path management.
  - `numpy`: For numerical calculations.
  - `matplotlib` and `FuncAnimation`: For real-time data visualization and animation.
  - `mpl_toolkits.mplot3d` and `Poly3DCollection`: For 3D visualizations.
  - `pandas`: For data handling and Excel file creation.
  - `openpyxl`: For working with Excel files.
  - `matplotlib.widgets.Slider`: For interactive sliders in visualization windows.

## Installation:
1. Clone this repository:
