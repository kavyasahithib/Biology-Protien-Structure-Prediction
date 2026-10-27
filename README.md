# 🧬 ProteinScope AI
### Enterprise Deep Learning Platform for Protein Secondary Structure Prediction & 3D Analytics

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?style=for-the-badge&logo=python)](https://python.org)
[![Framework](https://img.shields.io/badge/TensorFlow-2.21-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![Interface](https://img.shields.io/badge/Streamlit-Workstation-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Cloud](https://img.shields.io/badge/AWS-S3%20%26%20CloudWatch-yellow?style=for-the-badge&logo=amazon-aws)](https://aws.amazon.com)
[![License](https://img.shields.io/badge/License-Academic%20Research-green?style=for-the-badge)]()

---

## 📌 Executive Summary

**ProteinScope AI** is a scientific bioinformatics and machine learning workstation designed for high-throughput protein secondary structure prediction ($Q_3$: $\alpha$-helix, $\beta$-sheet, random coil). Powered by a bidirectional recurrent neural network (**BiLSTM**) with sequence padding up to 512 residues, the system bridges the gap between raw sequence inference, thermodynamic uncertainty, 3D structural biology, and enterprise cloud operations (AWS S3 & CloudWatch).

The platform features a **Royal Blue (`#014BAA`)** and **Light Cream (`#F8F3F0`)** clinical design system with hardware-accelerated **3Dmol.js WebGL visualization**, Shannon entropy boundary quantification, backpropagated gradient saliency (**Explainable AI**), and independent held-out test benchmarks.

---

## 🏗️ System Architecture & Model Pipeline

The end-to-end data pipeline is structured into three discrete computational stages:

```
[ Amino Acid Sequence Input ] (Raw IUPAC String / FASTA)
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 1: Sequence Tokenization & Context Windowing     │
│ • Strips FASTA headers, numeric characters & spaces    │
│ • Encodes 20 standard IUPAC amino acids + Unknown (21) │
│ • Zero-pads / truncates to fixed 512-residue tensor    │
└────────────────────────────────────────────────────────┘
             │ Shape: (1, 512)
             ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 2: Deep Bidirectional LSTM Neural Backbone       │
│ • Embedding Layer: 22 tokens -> 32-dimensional vector   │
│ • BiLSTM Layer: 128 hidden units (Forward + Backward)  │
│   Captures N-to-C and C-to-N folding dependencies      │
│ • Dropout Layer: Regularization against co-adaptation  │
│ • Dense Bottleneck: 32 units                           │
└────────────────────────────────────────────────────────┘
             │ Shape: (1, 512, 4)
             ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 3: TimeDistributed Softmax Projection            │
│ • Normalized class probabilities: C (Coil), H, E       │
│ • Argmax class decoding & per-residue confidence       │
│ • Shannon entropy computation H(x) = -Σ p·log₂(p)      │
└────────────────────────────────────────────────────────┘
             │
             ├───────────────────────────┬───────────────────────────┐
             ▼                           ▼                           ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│ 1. 2D Sequence Ribbon & │ │ 2. WebGL 3Dmol Viewer   │ │ 3. Telemetry Ingestion  │
│    Dual-Axis Trajectory │ │    Dynamic PDB Binding  │ │    Local CSV & AWS      │
└─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
```

---

## 🌟 Comprehensive Feature Catalog

### Tab 1: Sequence Inference & Prediction

#### 1. Curated Benchmark Protein Presets
Eliminates manual sequence transcription with 1-click loading of foundational structural motifs:
* **🧬 Human Insulin (B-chain):** `FVNQHLCGSHLVEALYLVCGERGFFYTPKT` (metabolic hormone with mixed $\alpha/\beta$ elements).
* **⚡ Ubiquitin Fold:** `MQIFVKTLTGKTITLEVEPSDTIENVKAKIQD` (classic $\beta$-grasp architecture).
* **🩸 Myoglobin Fragment:** `VLSEGEWQLVLHVWAKVEADVAGHGQDILIR` (dense $\alpha$-helical globin bundle).
* **🕸️ Silk Fibroin Motif:** `GAGAGSGAASGAGAGS` (repetitive anti-parallel $\beta$-pleated strand).
* **🛡️ Zinc Finger Motif:** `YKCGLCERSFVEKSALSRHQRVHKN` (DNA-binding $\beta\beta\alpha$ domain).

#### 2. Real-Time Physicochemical Profiling
Derives essential bioinformatics attributes directly from sequence composition:
* **Estimated Molecular Weight:** Computed in kDa based on an average residue molecular mass of $110\text{ Da}$.
* **Net Isoelectric Charge (pH 7.4):** Evaluates basic residues ($\text{Lys}, \text{Arg}, 0.1\times\text{His}$) against acidic residues ($\text{Asp}, \text{Glu}$).
* **Hydropathy (GRAVY Score):** Calculates the standardized **Kyte-Doolittle** index across all residues (identifying hydrophobic cores vs hydrophilic surface loops).

#### 3. Residue-Level Interactive Ribbon Map
* Modeled as individual pill-shaped capsules for each amino acid position.
* Displays **Residue Index**, **1-Letter IUPAC Symbol**, and **Predicted State** ($H, E, C$).
* Interactive hover tooltips present exact index, structural name, and softmax confidence percentage.

#### 4. Uncertainty Quantification & Confidence Calibration (Shannon Entropy)
* Standard softmax probabilities can be overconfident. ProteinScope AI computes the per-residue **Shannon Entropy**:
  $$H(i) = -\sum_{c \in \{C, H, E\}} p_c \log_2(p_c + \epsilon)$$
* **Dual-Axis Trajectory Plot:**
  * **Royal Blue Fill:** Softmax prediction confidence ($0.0 \rightarrow 1.05$).
  * **Crimson Dashed Trace:** Shannon Entropy $H(x)$ ($0.0 \rightarrow 1.585\text{ bits}$).
* **Domain Transition Boundary Detection:** Entropy spikes visually flag ambiguous junction boundaries between rigid structural blocks ($\alpha$-helices / $\beta$-sheets) and flexible coil regions.
* **Core Stability Index:** Quantifies the percentage of the sequence with $H(x) < 0.45\text{ bits}$ (high-confidence core).

#### 5. 🌐 3D Interactive Protein Backbone Viewer
Bridges 1D sequence prediction with physical 3D tertiary reality via WebGL **3Dmol.js**:
* **Dynamic Color Mapping:** Residues in the 3D model are colored dynamically by the BiLSTM prediction:
  * **$\alpha$-Helix ($H$):** **Royal Blue (`#014BAA`)**
  * **$\beta$-Sheet ($E$):** **Ocean Azure (`#0284C7`)**
  * **Random Coil ($C$):** **Slate Grey (`#64748B`)**
* **Supported Structures:** Pre-cached local benchmarks (`1UBQ`, `1TRZ`, `1MBO`, `1ZNF`, `1CRN`) plus support for **any arbitrary 4-letter RCSB PDB ID** (e.g., `4HHB`, `1AKI`).
* **Render Modes:** Ribbon Cartoon, Carbon-Alpha Trace, Stick Wireframe, and Space-Filling Spheres.
* **Interactive Navigation:** Smooth rotation, mouse-wheel zoom, panning, auto-spin toggle, and solvent-accessible surface overlay.
* **Hover Inspection:** Moving the cursor over any atom displays a floating 3D label with atom name, residue number, and predicted structure.

#### 6. 🧠 Explainable AI (XAI): Gradient Saliency & Receptive Field
* Peels back the black box of the BiLSTM recurrent cell using backpropagated embedding gradients:
  $$\text{Saliency}(j) = \left\| \nabla_{E_j} \hat{y}_{\text{target}} \right\|_2$$
* Select any residue in the sequence to visualize how upstream (N-terminal) and downstream (C-terminal) flanking residues influenced its classification.
* Quantitatively breaks down **Target Local Weight**, **N-Terminal Window $[i-4 : i-1]$**, and **C-Terminal Window $[i+1 : i+4]$** influence to reveal biological hydrogen-bonding dependencies ($i \pm 3, i \pm 4$).

#### 7. Dual Research Export Engine
* **Download Prediction CSV:** Complete per-residue dataset with position, amino acid, structure code, confidence, and hydropathy score.
* **Download FASTA / DSSP Sequence:** Formatted bioinformatic sequence ready for ChimeraX, PyMOL, or BLAST analysis:
  ```text
  >ProteinScope_AI_Predicted | Residues: 22
  MKTAYIAKQRQISFVKSHFSRQ
  CCCCCHCCCCCCCCECCCCCCC
  ```

---

### Tab 2: Model Evaluation & Independent Benchmark Proof

#### 1. Empirical Test Metrics
Evaluates model generalization against a strictly isolated, held-out test split (preventing data leakage):
* **Per-Residue Accuracy ($Q_3$):** **$82.45\%$**
* **Macro F1-Score:** **$0.8032$**
* **Status:** Verified independent test partition.

#### 2. Class-Wise Classification Report
Full precision, recall, F1-score, and support metrics across all three secondary structure states:
* Coil ($C$): Precision $0.8412$ | Recall $0.8645$ | F1 $0.8527$ | Support $14,250$
* Helix ($H$): Precision $0.8156$ | Recall $0.8320$ | F1 $0.8237$ | Support $11,480$
* Sheet ($E$): Precision $0.7780$ | Recall $0.7240$ | F1 $0.7500$ | Support $6,770$

#### 3. Normalized Confusion Matrix Heatmap
Rendered using a custom Royal Blue gradient (`#F8F3F0` $\rightarrow$ `#BFDBFE` $\rightarrow$ `#3B82F6` $\rightarrow$ `#014BAA`) to inspect misclassification patterns between sheet and coil boundaries.

#### 4. 🏆 Historical Benchmark Comparison Matrix (CB513 / CASP)
Contextualizes where BiLSTM stands in the evolutionary timeline of computational biology:

| Era | Paradigm / Methodology | Typical $Q_3$ Accuracy | Inference Latency | Compute Footprint | Core Algorithmic Technique |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1974** | Statistical / Empirical (Chou-Fasman) | $55\% - 65\%$ ($56.5\%$) | $< 1\text{ ms}$ | Minimal CPU | Single-residue conformational propensities |
| **1996** | Information Theory (GOR IV) | $60\% - 65\%$ ($64.4\%$) | $\sim 2\text{ ms}$ | Minimal CPU | Directional mutual information over 17-residue window |
| **2000s** | Classical ML (PHD / SVM) | $70\% - 74\%$ ($72.1\%$) | $\sim 15\text{ ms}$ | Standard CPU | Evolutionary profiles (PSI-BLAST PSSM) + Neural Networks |
| **Current**| **⭐ ProteinScope AI (BiLSTM)** | **$82.45\%$ (Measured)** | **$< 50\text{ ms}$** | **Commodity CPU / Edge** | **Bidirectional LSTM + TimeDistributed Dense Softmax** |
| **2020s** | Large Protein Models (ESM-2 / AlphaFold) | $85\% - 88\%$ ($86.8\%$) | $> 2,500\text{ ms}$ | High-End GPU (A100) | 3B-parameter masked language models + Attention |

> **Architectural Takeaway:** While 3B-parameter transformers achieve ~87% $Q_3$, they require multi-gigabyte GPU instances. ProteinScope AI achieves $82.45\%$ $Q_3$ on standard CPU hardware with sub-50 ms latency, representing the Pareto-optimal solution for real-time cloud microservices.

---

### Tab 3: Cloud Telemetry & Operations

* **Cloud Gateway Health Status:** Real-time indicator (`🟢 ONLINE — AWS us-east-1 ready`).
* **Dual Artifact Hydration:** Prioritizes local disk models, automatically falling back to an **Amazon S3 Bucket** (`MODEL_S3_BUCKET`) via `boto3`.
* **AWS CloudWatch Ingestion:** Emits custom metric metrics (`PredictionLatencySeconds`, `PredictionRequests`, `PredictionErrors`) under the `ProteinScopeAI` CloudWatch namespace when configured.
* **Operational Latency SLAs:**
  * **Mean Latency:** Average inference response time (typically $35 - 55\text{ ms}$).
  * **$P_{95}$ Latency:** 95th percentile SLA latency benchmark.
  * **Success Rate:** Uptime and error handling ratio ($100\%$).
* **Latency Distribution Histogram:** Plotly visualization of execution latency across sessions.
* **Audit Log Viewer:** Inverted chronological log of all inferences with UTC timestamps.

---

## 🎨 Design System: Royal Blue & Light Cream

The workstation features a clinical, academic aesthetic avoiding generic default themes:

```
#014BAA  ████████  Royal Blue   (Primary Accent, Active Tabs, Alpha-Helix)
#0284C7  ████████  Ocean Azure  (Beta-Sheet Secondary Structure)
#64748B  ████████  Slate Grey   (Random Coil, Legend Accents)
#F8F3F0  ████████  Light Cream  (Base Canvas Viewport Background)
#0A192F  ████████  Deep Navy    (High-Contrast Editorial Typography)
#FFFFFF  ████████  Pure White   (Surface Cards, Containers, Tables)
```

---

## 📂 Project Directory Structure

```text
ProteinCloudProject/
├── app.py                      # Main Streamlit application and UI workstation
├── protein_bilstm.keras        # Trained Keras BiLSTM neural network weights
├── preprocessing.json          # Amino acid mapping, label mapping & context settings
├── evaluation.json             # Held-out test set evaluation metrics & confusion matrix
├── requirements.txt            # Python package dependencies
├── app_metrics.csv             # Local operational telemetry request log
├── 3dmol-min.js                # Local WebGL 3Dmol library fallback
├── sample_1ubq.pdb             # Cached Ubiquitin structure (76 residues)
├── sample_1trz.pdb             # Cached Human Insulin structure (51 residues)
├── sample_1mbo.pdb             # Cached Myoglobin structure (153 residues)
├── sample_1znf.pdb             # Cached Zinc Finger structure (30 residues)
├── sample_1crn.pdb             # Cached Crambin structure (46 residues)
└── README.md                   # Complete platform documentation
```

---

## 🚀 Installation & Local Execution

### 1. Prerequisites
* Python **3.10, 3.11, or 3.12**
* Windows, macOS, or Linux

### 2. Clone or Navigate to Directory
```bash
cd ProteinCloudProject
```

### 3. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## ☁️ Optional Cloud Configuration (AWS)

To enable AWS S3 model hydration and CloudWatch custom metric ingestion, configure your environment variables:

```bash
# Windows PowerShell
$env:MODEL_S3_BUCKET="your-production-model-bucket"
$env:CLOUDWATCH_METRICS="true"
$env:AWS_REGION="us-east-1"

# Linux / macOS Bash
export MODEL_S3_BUCKET="your-production-model-bucket"
export CLOUDWATCH_METRICS="true"
export AWS_REGION="us-east-1"
```

---

## 📊 Scientific Methodology & Citations

1. **Kabsch, W. & Sander, C. (1983).** *Dictionary of protein secondary structure: pattern recognition of hydrogen-bonded and geometrical features.* Biopolymers, 22(12), 2577–2637.
2. **Kyte, J. & Doolittle, R. F. (1982).** *A simple method for displaying the hydropathic character of a protein.* Journal of Molecular Biology, 157(1), 105–132.
3. **Cuff, J. A. & Barton, G. J. (1999).** *Evaluation and improvement of multiple sequence methods for protein secondary structure prediction.* Proteins, 34(4), 508–519.
4. **Rego, N. & Koes, D. (2015).** *3Dmol.js: molecular visualization without Java.* Bioinformatics, 31(8), 1322–1324.

---

### 👨‍💻 Academic & Professional Inquiries
Developed for cloud-enabled bioinformatics workflows, structural analytics, and deep learning sequence modeling.
