"""
Genera el dataset de entrenamiento a partir de las salidas del solver MATLAB.

Flujo previsto (Fase 0-1 del plan):
1. En MATLAB, barrer el espacio de parametros definido (frecuencia, propiedades
   del medio Kelvin-Voigt, radio inicial de burbuja, etc.) y exportar cada
   simulacion a data/raw/ (formato .mat o .csv, uno por simulacion o un unico
   fichero con todas).
2. Este script lee esos ficheros, extrae inputs (parametros) y outputs
   (la magnitud objetivo v1, p.ej. amplitud/resonancia o curva de atenuacion),
   y construye un dataset tabular listo para PyTorch.
3. Guarda el resultado en data/processed/ (p.ej. como .npz o .csv), separado
   ya en train/val/test.

Por ahora esto es un esqueleto: falta decidir el formato exacto de exportacion
de MATLAB antes de rellenar la logica real (ver Fase 0 del plan en el
proyecto de Claude).
"""

from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_raw_simulations(raw_dir: Path = RAW_DIR):
    """Lee todas las simulaciones exportadas desde MATLAB en raw_dir.

    TODO: implementar una vez decidido el formato de exportacion
    (.mat via scipy.io.loadmat, o .csv via pandas).
    """
    raise NotImplementedError("Pendiente: definir formato de exportacion de MATLAB (Fase 0)")


def build_dataset(simulations):
    """Convierte la lista de simulaciones en arrays de inputs/outputs listos
    para entrenar (numpy o torch tensors).
    """
    raise NotImplementedError


def save_dataset(inputs, outputs, out_dir: Path = PROCESSED_DIR):
    """Guarda el dataset procesado (y hace el split train/val/test)."""
    raise NotImplementedError


if __name__ == "__main__":
    sims = load_raw_simulations()
    X, y = build_dataset(sims)
    save_dataset(X, y)
