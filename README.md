
# 🧬 Protein Dynamics Analysis Using DynamicPDB

## Overview

This project explores the dynamic behavior of proteins by leveraging the **Dynamic Protein Data Bank (DynamicPDB)**—a comprehensive dataset encompassing approximately 12,600 proteins, each subjected to 1-microsecond all-atom molecular dynamics (MD) simulations. The dataset provides a rich suite of physical properties, including atomic velocities, forces, potential and kinetic energies, and temperature profiles, recorded at 1 picosecond intervals.

Our analysis focuses on the **1a62_A** protein, examining its conformational changes and dynamic properties using advanced computational tools.

## Dataset Details

- **Source**: [DynamicPDB GitHub Repository](https://github.com/fudan-generative-vision/dynamicPDB)
- **Webpage**: [DynamicPDB Project Page](https://fudan-generative-vision.github.io/dynamicPDB/)
- **Publication**: [Dynamic PDB: A New Dataset and a SE(3) Model Extension](https://arxiv.org/abs/2408.12413)

## Project Objectives

- Analyze the dynamic behavior of the 1a62_A protein using MD simulation data.
- Extract and interpret physical properties such as atomic velocities and forces.
- Visualize conformational changes over time.
- Lay the groundwork for integrating machine learning models to predict protein dynamics.

## Data Acquisition and Preparation

1. **Install Git LFS**:

   ```bash
   sudo apt-get install git-lfs
   git lfs install
   ```

2. **Clone the DynamicPDB Repository**:

   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/datasets/fudan-generative-vision/dynamicPDB.git dynamicPDB_raw
   ```

3. **Download Specific Protein Data (e.g., 1a62_A)**:

   ```bash
   cd dynamicPDB_raw
   git lfs pull --include="1a62_A/*"
   ```

4. **Merge and Extract the Dataset**:

   ```bash
   cat 1a62_A/1a62_A.tar.gz.part* > 1a62_A/1a62_A.tar.gz
   mkdir ../dynamicPDB
   tar -xvzf 1a62_A/1a62_A.tar.gz -C ../dynamicPDB
   ```

## Directory Structure

```
dynamicPDB/
└── 1a62_A/
    ├── 1a62_A.pdb
    ├── 1a62_A.dcd
    ├── 1a62_A.pkl
    ├── 1a62_A_minimized.pdb
    ├── 1a62_A_nvt_equi.dat
    ├── 1a62_A_npt_equi.dat
    ├── 1a62_A_T.dcd
    ├── 1a62_A_T.pkl
    ├── 1a62_A_F.pkl
    ├── 1a62_A_V.pkl
    └── 1a62_A_state_npt100000.0.xml
```

## Tools and Libraries

- [MDAnalysis](https://www.mdanalysis.org/): For analyzing molecular dynamics trajectories.
- [NumPy](https://numpy.org/): For numerical computations.
- [Matplotlib](https://matplotlib.org/): For data visualization.
- [PyTorch](https://pytorch.org/): For potential integration of machine learning models.

## Sample Code Snippet

```python
import MDAnalysis as mda

# Load the protein structure and trajectory
u = mda.Universe("1a62_A.pdb", "1a62_A.dcd")

# Access atom positions at the first frame
positions = u.atoms.positions
print(positions)
```

## Potential Applications

- Understanding protein conformational changes.
- Predicting protein-ligand interactions.
- Developing machine learning models for protein dynamics.
- Contributing to drug discovery and design.

## References

- Liu, C., Wang, J., Cai, Z., et al. (2024). *Dynamic PDB: A New Dataset and a SE(3) Model Extension by Integrating Dynamic Behaviors and Physical Properties in Protein Structures*. [arXiv:2408.12413](https://arxiv.org/abs/2408.12413)
- DynamicPDB GitHub Repository: [https://github.com/fudan-generative-vision/dynamicPDB](https://github.com/fudan-generative-vision/dynamicPDB)
- DynamicPDB Project Page: [https://fudan-generative-vision.github.io/dynamicPDB/](https://fudan-generative-vision.github.io/dynamicPDB/)
