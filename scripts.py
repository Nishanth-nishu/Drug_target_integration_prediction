import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import optuna
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
from MDAnalysis.analysis.msd import EinsteinMSD  # Updated import

# --------------------------
# Advanced MD Analysis
# --------------------------
class AdvancedMDAnalysis:
    def __init__(self, pdb_file, trajectory_file):
        self.universe = mda.Universe(pdb_file, trajectory_file)
        
    def calculate_fel(self, cv1, cv2, bins=100):
        """Calculate 2D Free Energy Landscape using collective variables"""
        hist, xedges, yedges = np.histogram2d(cv1, cv2, bins=bins)
        prob = hist / hist.sum()
        fel = -np.log(np.where(prob > 0, prob, 1e-10))
        return (fel - fel.min()) / (fel.max() - fel.min())

    def analyze_hydrogen_bonds(self):
        """Calculate hydrogen bond statistics"""
        hbonds = HydrogenBondAnalysis(self.universe)
        hbonds.run()
        return hbonds.count_by_time()

    def calculate_diffusion(self):
        """Compute diffusion coefficients"""
        # Ensure that the trajectory is unwrapped before this step
        msd = EinsteinMSD(self.universe, select="type O", msd_type="xyz", fft=True)
        msd.run()
        return msd.results.timeseries

# --------------------------
# GNN for Molecular Properties
# --------------------------
class MolecularGNN(pl.LightningModule):
    def __init__(self, hidden_dim=128, num_layers=3, lr=1e-3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(10, hidden_dim))  # Input features should match your atom features
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        self.fc = nn.Linear(hidden_dim, 1)
        self.lr = lr
        self.loss_fn = nn.MSELoss()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, batch.y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# --------------------------
# Data Preparation
# --------------------------
def create_molecular_graphs(universe, n_frames=100):
    """Convert MD trajectory to graph dataset"""
    graphs = []
    for ts in universe.trajectory[:n_frames]:
        # Simple node features: atom types (example using random features)
        node_features = np.random.rand(len(universe.atoms), 10)  # Replace with real features
        
        # Create edge indices (distance-based)
        pos = universe.atoms.positions
        dist_matrix = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
        edge_index = np.argwhere(dist_matrix < 5.0)  # 5Å cutoff
        edge_index = torch.tensor(edge_index.T, dtype=torch.long)
        
        # Create synthetic target (example property)
        target = torch.tensor([np.mean(dist_matrix)], dtype=torch.float)
        
        graphs.append(Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            y=target
        ))
    return graphs

# --------------------------
# Hyperparameter Optimization
# --------------------------
def optimize_hyperparams(trial):
    return {
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 2, 5),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    }

# --------------------------
# Visualization Components
# --------------------------
class MDVisualizer:
    @staticmethod
    def plot_fel(fel, xlabel='CV1', ylabel='CV2'):
        plt.figure(figsize=(10, 8))
        sns.heatmap(fel.T, cmap='viridis')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Free Energy Landscape")
        plt.show()

    @staticmethod
    def plot_hbond_timeseries(hbond_counts):
        plt.figure(figsize=(10, 6))
        plt.plot(hbond_counts)
        plt.xlabel("Frame")
        plt.ylabel("Hydrogen Bonds")
        plt.title("Hydrogen Bond Dynamics")
        plt.show()

    @staticmethod
    def plot_diffusion(msd_timeseries):
        plt.figure(figsize=(10, 6))
        plt.plot(msd_timeseries)
        plt.xlabel("Time (ps)")
        plt.ylabel("MSD (Å²)")
        plt.title("Mean Square Displacement")
        plt.show()

# --------------------------
# Main Workflow
# --------------------------
def main(pdb_file, trajectory_file):
    # Advanced MD Analysis
    md_analyzer = AdvancedMDAnalysis(pdb_file, trajectory_file)
    
    # FEL Example
    cv1 = np.random.normal(size=1000)
    cv2 = np.random.normal(size=1000)
    fel = md_analyzer.calculate_fel(cv1, cv2)
    MDVisualizer.plot_fel(fel)

    # Hydrogen Bond Analysis
    hbonds = md_analyzer.analyze_hydrogen_bonds()
    MDVisualizer.plot_hbond_timeseries(hbonds)

    # Diffusion Analysis
    msd_timeseries = md_analyzer.calculate_diffusion()
    MDVisualizer.plot_diffusion(msd_timeseries)

    # Prepare GNN Dataset
    graphs = create_molecular_graphs(md_analyzer.universe)
    train_loader = DataLoader(graphs, batch_size=32, shuffle=True)

    # Hyperparameter Optimization
    def objective(trial):
        params = optimize_hyperparams(trial)
        model = MolecularGNN(**params)
        trainer = pl.Trainer(
            max_epochs=50,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False
        )
        trainer.fit(model, train_loader)
        return trainer.callback_metrics["train_loss"].item()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)  # Reduced trials for testing
    print("Best hyperparameters:", study.best_params)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Advanced MD/ML Analysis")
    parser.add_argument("--pdb", required=True, help="Input PDB file")
    parser.add_argument("--traj", required=True, help="Trajectory file")
    args = parser.parse_args()
    
    main(args.pdb, args.traj)
