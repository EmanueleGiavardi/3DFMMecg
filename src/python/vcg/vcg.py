import numpy as np

class VCG:
    """
    Rappresenta un Vectorcardiogramma (VCG) 3D.
    """
    def __init__(self, vcg_dict: dict = None, X: np.ndarray = None, Y: np.ndarray = None, Z: np.ndarray = None):
        """
        Inizializza l'oggetto VCG. Può essere inizializzato con un dizionario
        contenente 'X', 'Y', 'Z', oppure passando esplicitamente gli array.
        """
        if vcg_dict is not None:
            self._data = {
                'X': vcg_dict.get('X', np.array([])),
                'Y': vcg_dict.get('Y', np.array([])),
                'Z': vcg_dict.get('Z', np.array([]))
            }
        elif X is not None and Y is not None and Z is not None:
            self._data = {'X': X, 'Y': Y, 'Z': Z}
        else:
            raise ValueError("fornire un dizionario oppure gli array X, Y, Z esplicitamente.")

    @property
    def X(self) -> np.ndarray:
        return self._data['X']

    @X.setter
    def X(self, value: np.ndarray):
        self._data['X'] = value

    @property
    def Y(self) -> np.ndarray:
        return self._data['Y']

    @Y.setter
    def Y(self, value: np.ndarray):
        self._data['Y'] = value

    @property
    def Z(self) -> np.ndarray:
        return self._data['Z']

    @Z.setter
    def Z(self, value: np.ndarray):
        self._data['Z'] = value

    def to_array(self) -> np.ndarray:
        """
        Restituisce il VCG come array NumPy di forma (N, 3).
        
        Returns:
            np.ndarray: Array Nx3 dove le colonne sono rispettivamente X, Y, Z.
        """
        return np.column_stack((self.X, self.Y, self.Z))
        
    def to_dict(self) -> dict:
        """
        Restituisce il VCG come dizionario grezzo.
        """
        return self._data.copy()

    def __getitem__(self, key: str) -> np.ndarray:
        """
        Consente l'accesso tramite parentesi quadre come se fosse un dizionario (es. vcg['X']).
        Aggiunto per retrocompatibilità.
        """
        return self._data[key]

    def __setitem__(self, key: str, value: np.ndarray):
        if key not in ['X', 'Y', 'Z']:
            raise KeyError("Le chiavi valide sono 'X', 'Y', 'Z'")
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __repr__(self) -> str:
        shape = self.X.shape[0] if self.X is not None else 0
        return f"<VCG object (N={shape} samples)>"
