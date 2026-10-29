# 🧬 ProteinScope AI — Setup & Quickstart Guide

This guide contains step-by-step instructions to run **ProteinScope AI** locally on Windows, macOS, or Linux.

---

## 📌 Prerequisites
- **Python Version**: **Python 3.10, 3.11, or 3.12** *(Python 3.12 recommended)*.
  - Download from: https://www.python.org/downloads/
  - **Windows Users**: During installation, ensure you check the box: **"Add python.exe to PATH"**.
- **Disk Space**: ~2 GB free (for Python environment and dependencies).
- **Internet**: Required only during the initial dependency installation.

---

## 💻 Windows Setup Instructions

### 1. Open Terminal in Project Folder
Open this folder in File Explorer, click the address bar at the top, type `powershell` (or `cmd`), and hit **Enter**.

### 2. Create Virtual Environment
```powershell
python -m venv venv
```

### 3. Activate Virtual Environment
In PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```
*(If using Command Prompt `cmd.exe`: `venv\Scripts\activate.bat`)*

You should see `(venv)` appear at the beginning of the command prompt line.

### 4. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
*(This installs TensorFlow, Streamlit, Plotly, Pandas, and NumPy. It typically takes 2–3 minutes).*

### 5. Start the Application
```powershell
streamlit run app.py
```

The web application will open automatically in your browser at:
**http://localhost:8501**

To stop the server, press `Ctrl + C` in the terminal.

---

## 🍎 macOS / Linux Setup Instructions

### 1. Open Terminal in Project Folder
```bash
cd /path/to/ProteinScopeAI
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the Application
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🧪 Quick Functional Test
Once the interface loads:
1. Under **🧬 Sequence Inference** (Tab 1), click any preset button (e.g. **🧬 Human Insulin (B-chain)**).
2. Click **Analyze Protein Structure**.
3. Confirm that:
   - Secondary structure predictions ($H = \text{Helix}$, $E = \text{Sheet}$, $C = \text{Coil}$) display on the 2D ribbon.
   - The Shannon Entropy confidence graph renders.
   - The 3D WebGL molecular viewer renders the 3D protein backbone.
4. Click **📊 Model Evaluation & Benchmark** (Tab 2) to view independent held-out test benchmarks ($Q_3 = 82.45\%$).

---

## ❓ Common Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **`scripts cannot be executed` in PowerShell** | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then reactivate `.\venv\Scripts\Activate.ps1`. |
| **`python` command not found** | Re-run Python installer, choose **Modify**, and check **Add Python to environment variables**. |
| **TensorFlow install fails on Python 3.13** | Python 3.13 is too new for pre-built TensorFlow binary wheels. Install Python 3.12 from python.org. |
| **Port 8501 already in use** | Run on an alternative port: `streamlit run app.py --server.port 8502`. |

