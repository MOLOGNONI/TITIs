import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor


def advanced_visualizations(df):
    """
    Creates a suite of advanced visualizations to explore the TITIs dataset.

    This function generates multiple plots to provide graphical insights into
    the relationships between experimental parameters, matrix effects, and
    recovery. Plots include heatmaps, violin plots, and 3D response surfaces.

    Args:
        df (pd.DataFrame): The input dataframe containing experimental data.
    """
    print("\n=== Advanced Visualizations ===\n")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Matrix effect by technique and dilution
    pivot_me = df.pivot_table(
        values='matrix_effect',
        index='technique',
        columns='dilution_factor',
        aggfunc='mean'
    )
    sns.heatmap(pivot_me, annot=True, fmt='.1f', ax=axes[0, 0], cmap='RdYlBu_r')
    axes[0, 0].set_title('Matrix Effect by Technique and Dilution')

    # Recovery by technique and voltage
    df['voltage_bin'] = pd.cut(
        df['spray_voltage'], bins=5,
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    pivot_rec = df.pivot_table(
        values='recovery',
        index='technique',
        columns='voltage_bin',
        aggfunc='mean'
    )
    sns.heatmap(pivot_rec, annot=True, fmt='.3f', ax=axes[0, 1], cmap='YlOrRd')
    axes[0, 1].set_title('Recovery by Technique and Voltage')

    # Distributions by technique
    axes[1, 0].violinplot(
        [df[df['technique'] == tech]['matrix_effect'].values
         for tech in df['technique'].unique()],
        positions=range(len(df['technique'].unique()))
    )
    axes[1, 0].set_xticks(range(len(df['technique'].unique())))
    axes[1, 0].set_xticklabels(df['technique'].unique(), rotation=45)
    axes[1, 0].set_ylabel('Matrix Effect (%)')
    axes[1, 0].set_title('Matrix Effect Distribution')

    # 3D cluster analysis (projected in 2D)
    X_cluster = df[['matrix_effect', 'recovery', 'is_response_ratio']].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_cluster)
    scatter = axes[1, 1].scatter(
        X_cluster['matrix_effect'], X_cluster['recovery'],
        c=clusters, cmap='viridis', alpha=0.6
    )
    axes[1, 1].set_xlabel('Matrix Effect (%)')
    axes[1, 1].set_ylabel('Recovery')
    axes[1, 1].set_title('Analytical Behavior Clusters')
    plt.colorbar(scatter, ax=axes[1, 1])
    plt.tight_layout()
    plt.show()

    # Response surface analysis
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    x = df['spray_voltage']
    y = df['dilution_factor']
    z = df['matrix_effect']
    ax1.scatter(x, y, z, c=z, cmap='RdYlBu_r', alpha=0.6)
    ax1.set_xlabel('Spray Voltage')
    ax1.set_ylabel('Dilution Factor')
    ax1.set_zlabel('Matrix Effect')
    ax1.set_title('Response Surface - Matrix Effect')

    ax2 = fig.add_subplot(132, projection='3d')
    z2 = df['recovery']
    ax2.scatter(x, y, z2, c=z2, cmap='YlOrRd', alpha=0.6)
    ax2.set_xlabel('Spray Voltage')
    ax2.set_ylabel('Dilution Factor')
    ax2.set_zlabel('Recovery')
    ax2.set_title('Response Surface - Recovery')

    ax3 = fig.add_subplot(133)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df[[
        'spray_voltage', 'capillary_temp', 'nebulizer_flow',
        'collision_energy', 'matrix_effect', 'recovery'
    ]].dropna())
    colors_tech = {'SLE-LTP': 'red', 'QuEChERS': 'blue', 'SPE-OASIS': 'green',
                   'SPE-MIP': 'orange', 'PLE-EDGE': 'purple'}
    df_clean = df.dropna(subset=[
        'spray_voltage', 'capillary_temp', 'nebulizer_flow',
        'collision_energy', 'matrix_effect', 'recovery'
    ])

    for technique in df_clean['technique'].unique():
        mask = df_clean['technique'] == technique
        if mask.sum() > 0:
            ax3.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                label=technique, color=colors_tech[technique], alpha=0.7
            )
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax3.set_title('PCA - Separation by Technique')
    ax3.legend()
    plt.tight_layout()
    plt.show()


def generate_insights(df, correlation_matrix, significant_correlations):
    """
    Generates and prints a text-based summary of insights and recommendations.

    This function synthesizes the results from various statistical analyses
    to provide automated interpretations. It identifies strong correlations,
    summarizes performance by technique, determines the most influential
    variables using a RandomForest model, and suggests operational
    recommendations.

    Args:
        df (pd.DataFrame): The main dataframe with experimental data.
        correlation_matrix (pd.DataFrame): The correlation matrix from
                                           analyze_variable_dependencies.
        significant_correlations (list): A list of significant correlations
                                         from analyze_variable_dependencies.
    """
    print("\n=== AUTOMATIC INSIGHTS AND INTERPRETATION ===\n")
    print("1. DISCOVERED CORRELATION PATTERNS:")
    print("-" * 40)

    strong_correlations = [corr for corr in significant_correlations if abs(corr['correlation']) > 0.7]
    if strong_correlations:
        print("Strong Correlations (|r| > 0.7):")
        for corr in strong_correlations:
            direction = "positive" if corr['correlation'] > 0 else "negative"
            print(f"• {corr['var1']} ↔ {corr['var2']}: {direction} ({corr['correlation']:.3f})")

    print("\n2. ANALYSIS BY EXTRACTION TECHNIQUE:")
    print("-" * 40)
    for technique in df['technique'].unique():
        subset = df[df['technique'] == technique]
        print(f"\n{technique}:")
        print(f"  Average Matrix Effect: {subset['matrix_effect'].mean():.1f}% ± {subset['matrix_effect'].std():.1f}")
        print(f"  Average Recovery: {subset['recovery'].mean():.3f} ± {subset['recovery'].std():.3f}")
        print(f"  IS Response Ratio: {subset['is_response_ratio'].mean():.3f} ± {subset['is_response_ratio'].std():.3f}")

    print("\n3. DILUTION EFFECTS:")
    print("-" * 40)
    print("Dilution Factor vs Matrix Effect:")
    for dilution, group in df.groupby('dilution_factor'):
        print(f"  {dilution}x: ME = {group['matrix_effect'].mean():.1f}%, "
              f"Recovery = {group['recovery'].mean():.3f}")

    print("\n4. MOST INFLUENTIAL VARIABLES:")
    print("-" * 40)
    feature_cols = [
        'spray_voltage', 'capillary_temp', 'nebulizer_flow',
        'collision_energy', 'cleanup_power', 'specificity'
    ]
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df['matrix_effect'].fillna(df['matrix_effect'].mean())

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importance = sorted(list(zip(feature_cols, rf.feature_importances_)), key=lambda x: x[1], reverse=True)

    print("Importance for Matrix Effect prediction:")
    for feature, importance in feature_importance:
        print(f"  {feature}: {importance:.3f}")

    print("\n5. OPERATIONAL RECOMMENDATIONS:")
    print("-" * 40)
    best_technique_analysis = df.groupby('technique').agg({
        'matrix_effect': lambda x: abs(x).mean(),
        'recovery': 'mean',
        'is_response_ratio': lambda x: abs(x - 1).mean()
    })
    best_technique_analysis['score'] = (
        best_technique_analysis['matrix_effect'] +
        (1 - best_technique_analysis['recovery']) * 100 +
        best_technique_analysis['is_response_ratio'] * 100
    )
    best_tech = best_technique_analysis['score'].idxmin()

    print(f"• Recommended technique: {best_tech}")
    print("  (Lowest absolute matrix effect, best recovery, most stable IS response)")

    optimal_voltage = df.loc[df['matrix_effect'].abs().idxmin(), 'spray_voltage']
    optimal_temp = df.loc[df['matrix_effect'].abs().idxmin(), 'capillary_temp']
    print("• Suggested instrumental parameters:")
    print(f"  Spray Voltage: ~{optimal_voltage:.0f}V")
    print(f"  Capillary Temperature: ~{optimal_temp:.0f}°C")

    optimal_dilution = df.groupby('dilution_factor').agg(
        {'matrix_effect': lambda x: abs(x).mean()}
    )['matrix_effect'].idxmin()
    print(f"• Recommended dilution factor: {optimal_dilution}x")
    print("  (Minimizes matrix effect while maintaining sensitivity)")
