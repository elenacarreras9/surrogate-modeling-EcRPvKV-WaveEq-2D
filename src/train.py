"""
Bucle de entrenamiento del surrogate model.

Pendiente de completar en la Fase 2 del plan, una vez exista el dataset
procesado (ver generate_dataset.py) y este decidido el target v1 exacto.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import SurrogateMLP

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_processed_dataset(processed_dir: Path = PROCESSED_DIR):
    """Carga el dataset ya procesado (train/val) generado por generate_dataset.py.

    TODO: implementar una vez exista data/processed/*.npz (o el formato que se elija).
    """
    raise NotImplementedError


def train(model, train_loader, val_loader, n_epochs: int = 100, lr: float = 1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                pred = model(x_batch)
                val_loss += loss_fn(pred, y_batch).item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:4d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    return model


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_processed_dataset()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = SurrogateMLP(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
    train(model, train_loader, val_loader)
