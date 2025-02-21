# CETGNN Setup and Execution Guide

This guide will walk you through the steps to set up and run the GraphXAI project from the MIMS Harvard repository.

## Step-by-Step Instructions
### 1) Clone the GraphXAI Repository
```
# Clone the GraphXAI Repository
git clone https://github.com/mims-harvard/GraphXAI.git
```
### 2) Create a Python Environment
```
python -m venv myenv
```
### 3) Activate the Environment
```
# On Windows:
.\myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate
```
### 4) Install Requirements
```
pip install -r requirements.txt
```
### 5) Install DGL
```
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```
### 6) Change Directory to cetgnn
```
cd cetgnn
```
### 7) Create Folders
We demonstrate how to run the script for the FluorideCarbonyl dataset
```
mkdir form_1
```
### 8) Run the Script
```
python driver_main.py
```
