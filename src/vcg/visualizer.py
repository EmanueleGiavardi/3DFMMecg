import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np

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

    def animate_3d(self, step: int = 5):
        """
        Genera un'animazione 3D interattiva del VCG.
        """
        # step = ogni quanti frame visualizzare un punto nell'animazione
        first_wave = list(self.vcg_waves.values())[0]
        n_points = len(first_wave['X'])
        
        frame_indices = list(range(0, n_points, step))
        if frame_indices[-1] != n_points - 1:
            frame_indices.append(n_points - 1)
            
        fig = go.Figure()

        for i, wave_data in self.vcg_waves.items():
            color = self.colors[i % len(self.colors)]
            label = self.labels[i] if self.labels is not None else f'Onda {i+1}'
            
            fig.add_trace(go.Scatter3d(
                x=[wave_data['X'][0]],
                y=[wave_data['Y'][0]],
                z=[wave_data['Z'][0]],
                mode='lines',
                line=dict(color=color, width=4),
                name=label
            ))

        all_x = np.concatenate([w['X'] for w in self.vcg_waves.values()])
        all_y = np.concatenate([w['Y'] for w in self.vcg_waves.values()])
        all_z = np.concatenate([w['Z'] for w in self.vcg_waves.values()])
        
        def pad(vmin, vmax, pct=0.1):
            rng = vmax - vmin if vmax != vmin else 1
            return [vmin - rng*pct, vmax + rng*pct]

        fig.update_layout(
            height=700,
            scene=dict(
                xaxis=dict(range=pad(all_x.min(), all_x.max()), title='Asse X'),
                yaxis=dict(range=pad(all_y.min(), all_y.max()), title='Asse Y'),
                zaxis=dict(range=pad(all_z.min(), all_z.max()), title='Asse Z'),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1.5)
            ),
            title='VCG 3D Animato',
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 20, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="Pausa",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
                ]
            )]
        )

        frames = []
        for k in frame_indices:
            frame_data = []
            for i, wave_data in self.vcg_waves.items():
                frame_data.append(go.Scatter3d(
                    x=wave_data['X'][:k+1],
                    y=wave_data['Y'][:k+1],
                    z=wave_data['Z'][:k+1],
                    mode='lines'
                ))
            frames.append(go.Frame(data=frame_data, name=str(k)))
            
        fig.frames = frames
        return fig

    def animate_2d_planes(self, step: int = 5):
        """
        Genera un'animazione interattiva dei 3 piani 2D del VCG usando Plotly.
        """
        first_wave = list(self.vcg_waves.values())[0]
        n_points = len(first_wave['X'])
        
        frame_indices = list(range(0, n_points, step))
        if frame_indices[-1] != n_points - 1:
            frame_indices.append(n_points - 1)
            
        fig = sp.make_subplots(rows=1, cols=3, 
                               subplot_titles=('Piano Frontale (X, Y)', 'Piano Trasversale (X, Z)', 'Piano Sagittale (Z, Y)'))

        all_x = np.concatenate([w['X'] for w in self.vcg_waves.values()])
        all_y = np.concatenate([w['Y'] for w in self.vcg_waves.values()])
        all_z = np.concatenate([w['Z'] for w in self.vcg_waves.values()])
        
        def pad(vmin, vmax, pct=0.1):
            rng = vmax - vmin if vmax != vmin else 1
            return [vmin - rng*pct, vmax + rng*pct]

        for i, wave_data in self.vcg_waves.items():
            color = self.colors[i % len(self.colors)]
            label = self.labels[i] if self.labels is not None else f'Onda {i+1}'
            
            # frontale
            fig.add_trace(go.Scatter(x=[wave_data['X'][0]], y=[wave_data['Y'][0]], 
                                     mode='lines', line=dict(color=color, width=3), name=label, legendgroup=str(i)), 
                          row=1, col=1)
            # trasversale
            fig.add_trace(go.Scatter(x=[wave_data['X'][0]], y=[wave_data['Z'][0]], 
                                     mode='lines', line=dict(color=color, width=3), name=label, legendgroup=str(i), showlegend=False), 
                          row=1, col=2)
            # sagittale
            fig.add_trace(go.Scatter(x=[wave_data['Z'][0]], y=[wave_data['Y'][0]], 
                                     mode='lines', line=dict(color=color, width=3), name=label, legendgroup=str(i), showlegend=False), 
                          row=1, col=3)

        fig.update_xaxes(title_text="Asse X", range=pad(all_x.min(), all_x.max()), row=1, col=1)
        fig.update_yaxes(title_text="Asse Y", range=pad(all_y.min(), all_y.max()), row=1, col=1)
        
        fig.update_xaxes(title_text="Asse X", range=pad(all_x.min(), all_x.max()), row=1, col=2)
        fig.update_yaxes(title_text="Asse Z", range=pad(all_z.min(), all_z.max()), row=1, col=2)
        
        fig.update_xaxes(title_text="Asse Z", range=pad(all_z.min(), all_z.max()), row=1, col=3)
        fig.update_yaxes(title_text="Asse Y", range=pad(all_y.min(), all_y.max()), row=1, col=3)

        fig.update_layout(
            title='Proiezioni VCG 2D Animate',
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 20, "redraw": False}, "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="Pausa",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
                ]
            )]
        )

        frames = []
        for k in frame_indices:
            frame_data = []
            for i, wave_data in self.vcg_waves.items():
                frame_data.append(go.Scatter(x=wave_data['X'][:k+1], y=wave_data['Y'][:k+1], mode='lines'))
                frame_data.append(go.Scatter(x=wave_data['X'][:k+1], y=wave_data['Z'][:k+1], mode='lines'))
                frame_data.append(go.Scatter(x=wave_data['Z'][:k+1], y=wave_data['Y'][:k+1], mode='lines'))
            frames.append(go.Frame(data=frame_data, name=str(k)))
            
        fig.frames = frames
        return fig