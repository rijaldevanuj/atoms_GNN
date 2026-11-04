# ⚛️ Atoms GNN: Atomic Force & Energy Predictor

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-PyG-3DDC84.svg?logo=pytorch-lightning&logoColor=white)](https://pyg.org/)
[![Gradio UI](https://img.shields.io/badge/Gradio-Web%20Interface-FF5500.svg?logo=gradio&logoColor=white)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A Graph Neural Network (GNN) framework built with **PyTorch Geometric** to model 3D molecular graph representations, predicting quantum properties and 3D equilibrium forces directly from atomic structures without expensive Density Functional Theory (DFT) calculations.

---

## 📌 Project Overview

Traditional computational chemistry relies on Density Functional Theory (DFT) molecular dynamics simulations to settle atomic systems into their minimum-energy equilibrium configurations. However, DFT calculations scale heavily in computational expense.

**Atoms GNN** maps atomic structures into undirected spatial graphs, passing node and edge features through Message Passing Graph Convolutional Networks (`GCNConv`). The framework supports both graph-level global property prediction (e.g., internal energy $U_0$) and node-level local vector prediction (e.g., 3D forces $F_x, F_y, F_z$).

---

## 🔬 Model Architecture & Pipeline

```
  [ 3D Molecule / Structure ]
               │
               ▼
   [ Graph Representation ]
  ├── Nodes (v_i) : 11D / 92D Atomic Features (Group, Period, Radius, etc.)
  └── Edges (e_ij): Spatial Adjacency / Distance Cutoffs
               │
               ▼
   [ Message Passing Pipeline ]
  ├── Layer 1: GCNConv(in_features, 64) + ReLU
  └── Layer 2: GCNConv(64, 64)
               │
       ┌───────┴─────────────────────────┐
       ▼                                 ▼
[ Node-Level Head ]            [ Graph-Level Head ]
  No Pooling                     global_mean_pool
       │                                 │
  Linear(64, 3)                     Linear(64, 1)
       │                                 │
       ▼                                 ▼
(3D Atomic Forces: Fx,Fy,Fz)   (Global Quantum Property / Energy)
```

### Key Technical Phases
1. **Graph Construction**: Molecules are parsed into PyTorch Geometric `Data` objects containing node feature matrices $X \in \mathbb{R}^{N \times F}$ and Coordinate Format edge index tensors $E \in \mathbb{R}^{2 \times M}$.
2. **Message Passing & Aggregation**: Continuous local features are updated by aggregating representations from 1-hop and 2-hop spatial neighbors.
3. **Dual Prediction Heads**:
   - **Global Readout**: Uses `global_mean_pool` to aggregate all node vectors into a single graph vector for scalar properties.
   - **Local Node Head**: Omits pooling to project 64-dimensional node states into 3D directional vectors ($F_x, F_y, F_z$).

---

## 🚀 Features & Interactive Web GUI

- **Fast Inference**: Instant structural prediction bypassing iterative DFT steps.
- **Built-in Dataset Support**: Native integration with the **QM9 dataset** (130,000+ small organic molecules).
- **Interactive Web Interface**: Includes a **Gradio** web app to inspect and predict force vectors visually.

### Interactive Web Dashboard

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Atomic Force Predictor                          │
│                                                                        │
│   Input Molecule Index (0 to 1000):                                    │
│   [  42  ]  [ Submit ]                                                 │
│                                                                        │
│   GNN Output:                                                          │
│   ------------------------------------------------------------------   │
│   Analyzed Molecule 42 (9 atoms).                                      │
│   Predicted 3D Forces (Fx, Fy, Fz):                                    │
│   [[ 0.0412  -0.1284   0.0911]                                         │
│    [-0.0123   0.0541  -0.0332]                                         │
│    ...                                                                 │
│    [ 0.0891  -0.0024  -0.0712]]                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/rijaldevanuj/atoms_GNN.git
cd atoms_GNN
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Interactive Gradio Interface
```bash
python app.py
```
Open the generated local URL (`http://127.0.0.1:7860`) or public link in your browser.

---

## 📊 Dataset Details

The default implementation utilizes the **QM9 Dataset**:
- **Molecules**: ~134,000 organic structures composed of C, H, O, N, and F.
- **Properties**: 19 quantum chemical properties (Dipole Moment, Isotropic Polarizability, Internal Energy $U_0$, Homo/Lumo Energies, etc.).
- **Graph Dimensions**: ~9 to 29 nodes per graph with continuous 3D spatial coordinates.

---

## 📚 Academic Citations & References

If you build upon this project or reference it in your research, please cite the foundational literature on Graph Neural Networks in materials science:

1. **Crystal Graph Convolutional Neural Networks (CGCNN)**
   > Xie, T., & Grossman, J. C. (2018). *Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties*. Physical Review Letters, 120(14), 145301.
   > [DOI: 10.1103/PhysRevLett.120.145301](https://doi.org/10.1103/PhysRevLett.120.145301)

2. **SchNet Architecture for Atomistic Systems**
   > Schütt, K. T., Sauceda, H. E., Kindermans, P. J., Tkatchenko, A., & Müller, K. R. (2018). *SchNet – A deep learning architecture for molecules and materials*. The Journal of Chemical Physics, 148(24), 241722.
   > [DOI: 10.1063/1.5019779](https://doi.org/10.1063/1.5019779)

3. **QM9 Quantum Chemistry Benchmark Dataset**
   > Ramakrishnan, R., Dral, P. O., Rupp, M., & Von Lilienfeld, O. A. (2014). *Quantum chemistry structures and properties of 134 kilo molecules*. Scientific Data, 1, 140022.
   > [DOI: 10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22)

4. **Predicting Equilibrium States of Atomic Structures**
   > Yoon, J., Zhan, N., & Liu, M. *Convolutional Neural Network for Atomic Structures*. Carnegie Mellon University.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/rijaldevanuj/atoms_GNN/issues).

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
