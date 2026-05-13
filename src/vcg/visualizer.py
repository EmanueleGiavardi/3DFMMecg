import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VCGVisualizer:
    """
    Classe per rendering 3D e 2D del VCG.
    """

    def __init__(self, vcg_waves: dict, labels: list = None):
        """
        Costruttore
        
        Args:
            vcg_waves (dict): Dizionario con le traiettorie calcolate VCGCalculator.
            labels (list, optional): Etichette da assegnare alle diverse onde. Defaults to None.
        """
        self.vcg_waves = vcg_waves
        self.labels = labels
        self.colors = ['green', 'red', 'gold', 'blue', 'purple']

    def plot_3d(self, show: bool = True):
        """
        Renderizza una vista globale 3D del VCG.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, wave_data in self.vcg_waves.items():
            color = self.colors[i % len(self.colors)]
            label = self.labels[i] if self.labels is not None else f'Onda {i+1}'
            
            ax.plot(wave_data['X'], wave_data['Y'], wave_data['Z'], color=color, label=label, linewidth=2.5)
            
        ax.set_xlabel('Asse X (Destra-Sinistra)')
        ax.set_ylabel('Asse Y (Superiore-Inferiore)')
        ax.set_zlabel('Asse Z (Antero-Posteriore)')
        ax.set_title('VCG Decomposto (Modello 3D-FMM)')
        
        ax.view_init(elev=20., azim=-45)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        if show: plt.show()
            
        return fig

    def plot_2d_planes(self, show: bool = True):
        """
        Costruisce e renderizza le 3 proiezioni 2D del VCG.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, wave_data in self.vcg_waves.items():
            color = self.colors[i % len(self.colors)]
            label = self.labels[i] if self.labels is not None else f'Onda {i+1}'
            X, Y, Z = wave_data['X'], wave_data['Y'], wave_data['Z']
            
            # piano verticale
            axes[0].plot(X, Y, color=color, label=label, linewidth=2)
            
            # piano trasversale
            axes[1].plot(X, Z, color=color, linewidth=2)
            
            # piano sagittale
            axes[2].plot(Z, Y, color=color, linewidth=2)
            
        titles = ['Piano Frontale (X, Y)', 'Piano Trasversale (X, Z)', 'Piano Sagittale (Z, Y)']
        x_labels = ['Asse X', 'Asse X', 'Asse Z']
        y_labels = ['Asse Y', 'Asse Z', 'Asse Y']
        
        for ax, title, x_lb, y_lb in zip(axes, titles, x_labels, y_labels):
            ax.set_title(title)
            ax.set_xlabel(x_lb)
            ax.set_ylabel(y_lb)
            ax.grid(True, linestyle='--', alpha=0.6)
            
        axes[0].legend(loc='best')
        plt.tight_layout()
        if show:
            plt.show()
            
        return fig
