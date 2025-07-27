# ☀️ Solar Module Parameter Optimization using JAYA Algorithm

> Estimate key parameters of photovoltaic modules (ideality factor, series & parallel resistance) using a simple, parameter‑free JAYA optimizer.

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)  
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](#installation)  

---

## 📝 Table of Contents
- [✨ Features](#-features)  
- [🚀 Installation](#-installation)  
- [🎯 Usage](#-usage)  
- [📂 Project Structure](#-project-structure)  
- [🔬 Algorithm Details](#-algorithm-details)  
- [⚙️ Extending the Project](#-extending-the-project)  
- [📚 References](#-references)  
- [📝 License](#-license)  

---

## ✨ Features
- **JAYA Algorithm**: simple, parameter‑free metaheuristic  
- **Multi‑Module Support**: KC200GT, Shell SQ85, ST40  
- **Comprehensive Analysis**: multiple runs, convergence & I–V/P–V plots, 3D distributions  
- **Flexible Exports**: JSON, CSV, MATLAB formats  

---

## 🚀 Installation
```bash
git clone <repository-url>
cd solar_module_optimization
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🎯 Usage

### Basic
```bash
python main.py
```
Runs 30 trials on ST40, saves to `results/`, and shows plots.

### Custom
```python
from models.solar_module import SolarModuleModel
from optimization.jaya import JAYAOptimizer

model = SolarModuleModel('KC200GT', temperature=25)
opt = JAYAOptimizer(
    objective_func=model.objective_function,
    bounds={'a':[0.5,2.0],'Rs':[0.001,1.0],'Rp':[50,200]},
    population_size=50,
    max_iterations=1000
)
best_params, best_score = opt.optimize()
print(best_params, best_score)
```

### Analyzing Results
```python
from utils.data_handler import DataHandler
from utils.visualization import Visualizer

dh = DataHandler()
res = dh.load_results('results/JAYA_ST40_YYYYMMDD_HHMMSS.json')
viz = Visualizer()
viz.plot_iv_pv(res)
viz.plot_3d([res], ['JAYA'])
```

---

## 📂 Project Structure
```
solar_module_optimization/
├── main.py
├── requirements.txt
├── README.md
├── config/          # module specs
├── optimization/    # JAYA & BaseOptimizer
├── models/          # three‑diode model & objective
├── utils/           # plotting & I/O
└── results/         # outputs
```

---

## 🔬 Algorithm Details
**Update:**  
Xₙₑw = Xₒₗd + r₁·(X_best–|Xₒₗd|) – r₂·(X_worst–|Xₒₗd|)  
Minimize MSE at Voc (I=0), Isc (V=0), Vm & Im.

---

## ⚙️ Extending the Project

### Add Solar Module
In `config/solar_modules.py`:
```python
SOLAR_MODULES['NEW'] = {
  'name':'New','Voc':31.2,'Isc':7.9,'Vm':25.8,'Im':7.1,'Nc':60
}
```

### Add Optimizer
Create `optimization/my_opt.py`, subclass `BaseOptimizer`, implement `optimize()`.

---

## 📚 References
1. Rao, R. (2016). Jaya: A simple and new optimization algorithm. *IJIEC*, 7(1), 19–34.  
2. Tanabe & Fukunaga (2014). Improving SHADE via linear population size reduction. *IEEE CEC*.

---

## 📝 License
MIT © Your Name
