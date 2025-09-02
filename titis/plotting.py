import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def advanced_visualizations(df):
    """
    Creates advanced visualizations and graphical insights.
    """
    print("\n=== Visualizações Avançadas ===\n")

    # 1. Heatmap of interactions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Matrix effect by technique and dilution
    pivot_me = df.pivot_table(values='matrix_effect',
                             index='technique',
                             columns='dilution_factor',
                             aggfunc='mean')
    sns.heatmap(pivot_me, annot=True, fmt='.1f', ax=axes[0, 0], cmap='RdYlBu_r')
    axes[0, 0].set_title('Efeito Matriz por Técnica e Diluição')

    # Recovery by technique and voltage
    df['voltage_bin'] = pd.cut(df['spray_voltage'], bins=5, labels=['Muito Baixo', 'Baixo', 'Médio', 'Alto', 'Muito Alto'])
    pivot_rec = df.pivot_table(values='recovery',
                              index='technique',
                              columns='voltage_bin',
                              aggfunc='mean')
    sns.heatmap(pivot_rec, annot=True, fmt='.3f', ax=axes[0, 1], cmap='YlOrRd')
    axes[0, 1].set_title('Recovery por Técnica e Voltagem')

    # Distributions by technique
    axes[1, 0].violinplot([df[df['technique'] == tech]['matrix_effect'].values
                         for tech in df['technique'].unique()],
                        positions=range(len(df['technique'].unique())))
    axes[1, 0].set_xticks(range(len(df['technique'].unique())))
    axes[1, 0].set_xticklabels(df['technique'].unique(), rotation=45)
    axes[1, 0].set_ylabel('Matrix Effect (%)')
    axes[1, 0].set_title('Distribuição do Efeito Matriz')

    # 3D cluster analysis (projected in 2D)
    X_cluster = df[['matrix_effect', 'recovery', 'is_response_ratio']].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_cluster)
    scatter = axes[1, 1].scatter(X_cluster['matrix_effect'], X_cluster['recovery'],
                               c=clusters, cmap='viridis', alpha=0.6)
    axes[1, 1].set_xlabel('Matrix Effect (%)')
    axes[1, 1].set_ylabel('Recovery')
    axes[1, 1].set_title('Clusters de Comportamento Analítico')
    plt.colorbar(scatter, ax=axes[1, 1])
    plt.tight_layout()
    plt.show()

    # 2. Response surface analysis
    fig = plt.figure(figsize=(15, 5))

    # Subplot 1: Matrix Effect vs Parameters
    ax1 = fig.add_subplot(131, projection='3d')
    x = df['spray_voltage']
    y = df['dilution_factor']
    z = df['matrix_effect']
    ax1.scatter(x, y, z, c=z, cmap='RdYlBu_r', alpha=0.6)
    ax1.set_xlabel('Spray Voltage')
    ax1.set_ylabel('Dilution Factor')
    ax1.set_zlabel('Matrix Effect')
    ax1.set_title('Superfície de Resposta - Matrix Effect')

    # Subplot 2: Recovery vs Parameters
    ax2 = fig.add_subplot(132, projection='3d')
    z2 = df['recovery']
    ax2.scatter(x, y, z2, c=z2, cmap='YlOrRd', alpha=0.6)
    ax2.set_xlabel('Spray Voltage')
    ax2.set_ylabel('Dilution Factor')
    ax2.set_zlabel('Recovery')
    ax2.set_title('Superfície de Resposta - Recovery')

    # Subplot 3: PCA 2D analysis
    ax3 = fig.add_subplot(133)
    pca = PCA(n_components=2)
    pca_cols = ['spray_voltage', 'capillary_temp', 'nebulizer_flow',
                'collision_energy', 'matrix_effect', 'recovery']
    X_pca_data = df[pca_cols].dropna()
    X_pca = pca.fit_transform(X_pca_data)

    colors_tech = {'SLE-LTP': 'red', 'QuEChERS': 'blue', 'SPE-OASIS': 'green',
                   'SPE-MIP': 'orange', 'PLE-EDGE': 'purple'}

    y_pca = df.loc[X_pca_data.index, 'technique']

    for technique in y_pca.unique():
        mask = y_pca == technique
        ax3.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=technique, color=colors_tech[technique], alpha=0.7)

    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax3.set_title('PCA - Separação por Técnica')
    ax3.legend()
    plt.tight_layout()
    plt.show()
