# ApneaSense

## 📝 Project Description

Multi-Modal Spatial-Temporal Sleep Apnea Risk Screening

This project is developed as part of the Practise Module for the **Intelligent Sensing System Graduate Certification (MTech AIS) at NUS-ISS**.

### 👩‍💻 Contributors:

Arshi Saxena
Fia Thottan
John Joseph Peter

## 🛠️ Project Setup

This project uses **Python 3.12.7** and requires a virtual environment to manage dependencies. Follow these steps to set up the project locally:

### 1. Clone the repository
`git clone https://github.com/sensingproject07/ApneaSense.git`

### 2. Python Version
Ensure you have python version **Python 3.12.x** installed before proceeding with the next steps.

### 3. Run the setup script
`cd <your_project_path>/ApneaSense`

#### Windows
`.\setup_env.bat`

#### macOS/Linux
`bash setup_env.sh`

### 4. Activate the virtual environment

This project uses **VS Code auto-activation** as configured in `.vscode/settings.json`.  

- **If you are using VS Code:**  
  Ensure that Python extension is installed and enabled. Opening a new terminal(cmd) in VS Code for this project will automatically activate the virtual environment, and it will automatically deactivate when you close VS Code.
  **Check:** (.venv) should be appended to the path in cmd terminal in VS Code for successful auto-activation
  `(.venv) path\BiasTrack>`

- **If you are using any other IDE or terminal:**  
  You will need to manually activate the environment **each time** you open the project:

    #### Windows - Powershell
    `.venv\Scripts\activate.bat`

    #### macOS/Linux
    `source .venv/bin/activate`

### 5. Install PyTorch

This project involves training CNN-based deep learning models. While the code can run on CPU for debugging or lightweight checks, a **GPU-enabled PyTorch installation is strongly recommended for training**.

Each contributor is responsible for installing the PyTorch version appropriate for their own system. If you have a compatible NVIDIA GPU, install the CUDA-enabled version of PyTorch.

Please use the official PyTorch installation selector to get the correct command for your machine:  
`https://pytorch.org/get-started/locally/`