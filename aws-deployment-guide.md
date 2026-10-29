# ☁️ ProteinScope AI — AWS Cloud Deployment Guide (EC2 Free Tier)

Step-by-step production runbook to deploy **ProteinScope AI** on AWS EC2 Free Tier (`t2.micro` / `t3.micro`) using Ubuntu 24.04 LTS.

---

## 🏛️ Architecture Overview
```
[ User Browser ]
       │
       │ HTTP Port 8501 (Restricted to your Public IP: x.x.x.x/32)
       ▼
┌────────────────────────────────────────────────────────┐
│ Amazon EC2 (Ubuntu 24.04 LTS — t2.micro or t3.micro)   │
│                                                        │
│ • 2 GB Swap File (EBS) -> Prevents OOM crashes         │
│ • Python 3.12 Virtual Environment                      │
│ • systemd Background Service (proteinscope.service)    │
│ • Streamlit Web Server (Port 8501)                     │
│ • IAM Instance Profile (ProteinScopeEC2Role)           │
└──────────────────┬─────────────────┬───────────────────┘
                   │                 │
                   ▼                 ▼
          ┌─────────────────┐ ┌──────────────────────────┐
          │ Amazon S3       │ │ Amazon CloudWatch        │
          │ (Private Model  │ │ Namespace:               │
          │  Artifacts)     │ │ ProteinScopeAI           │
          └─────────────────┘ └──────────────────────────┘
```

---

## 💰 Billing & Free Tier Safety Guardrails
- **Instance Type**: Select `t2.micro` or `t3.micro` displaying the **"Free tier eligible"** green badge.
- **Storage**: Allocate a 15 GiB `gp3` root volume (under the 30 GB monthly Free Tier limit).
- **Public IPv4**: AWS charges ~$0.005/hour for in-use public IPv4 addresses (~$0.12 for a 24-hour demo). Remember to terminate the instance after testing.
- **Security**: Inbound ports 22 (SSH) and 8501 (Streamlit) must be restricted to **"My IP"** only. Never open port 8501 to `0.0.0.0/0`.

---

## 🚀 Step-by-Step Manual Deployment Instructions

### Step 1: Launch EC2 Instance (AWS Console)
1. Navigate to **EC2** -> **Instances** -> **Launch Instances**.
2. **Name**: `ProteinScope-Inference-Server`.
3. **AMI**: **Ubuntu Server 24.04 LTS (HVM)**, 64-bit (x86).
4. **Instance Type**: `t2.micro` or `t3.micro` (Free tier eligible).
5. **Key Pair**: Create new key pair `proteinscope-key` (RSA, `.pem`) and download it.
6. **Network Settings**:
   - Auto-assign public IP: **Enable**.
   - Security Group Rules:
     - **SSH**: Port 22 -> Source: **My IP**.
     - **Custom TCP**: Port 8501 -> Source: **My IP**.
7. **Storage**: `15 GiB` of `gp3`. Confirm **Delete on termination** is set to `Yes`.
8. Click **Launch Instance** and copy the **Public IPv4 address**.

---

### Step 2: Connect from Windows (PowerShell)
```powershell
# Set secure key permissions on Windows
cd C:\Users\sailo\.ssh
icacls proteinscope-key.pem /inheritance:r
icacls proteinscope-key.pem /grant:r "$($env:USERNAME):(R)"

# Connect to Ubuntu EC2
ssh -i "proteinscope-key.pem" ubuntu@<YOUR_EC2_PUBLIC_IP>
```

---

### Step 3: Configure System & Swap Memory (Inside Ubuntu SSH)
*TensorFlow on 1 GB RAM requires a 2 GB swap file to prevent out-of-memory (`Killed`) errors:*

```bash
# 1. Update OS and install dependencies
sudo apt update && sudo apt install -y python3-pip python3-venv git htop

# 2. Configure 2 GB Swap file
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 3. Clone repository from GitHub
cd ~
git clone https://github.com/kavyasahithib/Biology-Protien-Structure-Prediction.git proteinscope
cd proteinscope

# 4. Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: Configure systemd Background Service (Inside Ubuntu SSH)
```bash
# Copy and enable the service
sudo cp ~/proteinscope/proteinscope.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable proteinscope
sudo systemctl start proteinscope

# Check service status
sudo systemctl status proteinscope --no-pager
```

---

### Step 5: Test the Live Application
Open your browser and navigate to:
```
http://<YOUR_EC2_PUBLIC_IP>:8501
```
1. Under **🧬 Sequence Inference** (Tab 1), test the preset protein **🧬 Human Insulin (B-chain)**.
2. Confirm per-residue prediction, Shannon entropy graph, and 3D WebGL protein backbone.
3. Check **📊 Model Evaluation** (Tab 2) for the held-out test benchmark metrics ($Q_3 = 82.45\%$).

---

## 🧹 Complete Resource Teardown Checklist
To avoid unnecessary ongoing charges:
1. **EC2 Console** -> **Instances** -> Select `ProteinScope-Inference-Server` -> **Instance State** -> **Terminate instance**.
2. **EC2 Console** -> **Volumes** -> Confirm no detached EBS volumes remain.
3. **Billing Console** -> Review current month-to-date charges to confirm $0.00.

