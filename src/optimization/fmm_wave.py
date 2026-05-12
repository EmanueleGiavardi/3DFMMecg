from dataclasses import dataclass

@dataclass
class FMMWave:
    """Parametri di una singola onda FMM."""
    M: float # shift verticale dell'onda
    A: float # ampiezza dell'onda
    alpha: float # locazione temporale dell'onda
    beta: float # direzione di picco dell'onda
    omega: float # larghezza dell'onda
    variance_explained: float # varianza spiegata dall'onda