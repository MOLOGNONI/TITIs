import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from statsmodels.multivariate.manova import MANOVA
from statsmodels.multivariate.cancorr import CanCorr


warnings.filterwarnings('ignore')


def set_professional_style():
    """
    Sets a professional and visually appealing style for matplotlib and seaborn plots.

    This function updates the global `rcParams` to ensure all subsequent plots
    have a consistent and publication-quality look.
    """
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })


def analyze_variable_dependencies(df):
    """
    Analyzes and visualizes dependencies between numerical variables in the dataframe.

    This function calculates the Pearson correlation matrix for all numerical
    columns, displays it as a heatmap, and prints out correlations that
    exceed a significance threshold of |r| > 0.5.

    Args:
        df (pd.DataFrame): The input dataframe containing experimental data.

    Returns:
        tuple[pd.DataFrame, list]: A tuple containing:
            - correlation_matrix (pd.DataFrame): The full correlation matrix.
            - significant_correlations (list): A list of dictionaries, each
              detailing a pair of variables with a correlation > 0.5.
    """
    print("=== Variable Dependency Analysis ===\n")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix - TITIs Variables', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()

    print("Significant Correlations (|r| > 0.5):")
    significant_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                var1 = correlation_matrix.columns[i]
                var2 = correlation_matrix.columns[j]
                significant_correlations.append({
                    'var1': var1, 'var2': var2, 'correlation': corr_val
                })
                print(f"{var1} ↔ {var2}: {corr_val:.3f}")
    return correlation_matrix, significant_correlations


def perform_manova_analysis(df):
    """
    Performs a Multivariate Analysis of Variance (MANOVA).

    This analysis tests whether there are significant differences between the
    means of the extraction techniques ('technique') across multiple dependent
    variables ('matrix_effect', 'recovery', 'is_response_ratio') simultaneously.
    It also performs post-hoc univariate ANOVAs for each dependent variable.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The subset of the dataframe used for the MANOVA,
                      with NA values dropped.
    """
    print("\n=== MANOVA Analysis ===\n")
    dependent_vars = ['matrix_effect', 'recovery', 'is_response_ratio']
    independent_var = 'technique'
    data_manova = df[dependent_vars + [independent_var]].dropna()

    try:
        manova = MANOVA.from_formula(
            ' + '.join(dependent_vars) + ' ~ ' + independent_var,
            data=data_manova
        )
        manova_results = manova.mv_test()
        print("MANOVA Results:")
        print(manova_results)

        print("\n=== Univariate Tests (ANOVA) ===")
        for var in dependent_vars:
            print(f"\n{var}:")
            groups = [group[var].values for name, group in
                      data_manova.groupby(independent_var)]
            f_stat, p_value = stats.f_oneway(*groups)
            print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")
            if p_value < 0.001:
                print("*** Highly significant difference")
            elif p_value < 0.01:
                print("** Very significant difference")
            elif p_value < 0.05:
                print("* Significant difference")
            else:
                print("No significant difference")
    except Exception as e:
        print(f"Error in MANOVA: {e}")
        print("Performing individual ANOVAs...")
        for var in dependent_vars:
            print(f"\nANOVA for {var}:")
            groups = [group[var].values for name, group in
                      data_manova.groupby(independent_var)]
            f_stat, p_value = stats.f_oneway(*groups)
            print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, var in enumerate(dependent_vars):
        sns.boxplot(data=data_manova, x=independent_var, y=var, ax=axes[i])
        axes[i].set_title(f'{var} by Technique', fontsize=14)
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    return data_manova


def canonical_correlation_analysis(df):
    """
    Performs canonical correlation analysis (CCA).

    CCA finds the linear relationships between two sets of variables:
    Set 1: Extraction technique properties ('cleanup_power', etc.)
    Set 2: Instrumental parameters and results.
    It helps understand the underlying correlations between these two domains.
    If CCA fails, it falls back to a Principal Component Analysis (PCA).

    Args:
        df (pd.DataFrame): The input dataframe.
    """
    print("\n=== Canonical Correlation Analysis ===\n")
    set1_vars = ['cleanup_power', 'specificity', 'base_recovery', 'variability']
    set2_vars = [
        'spray_voltage', 'capillary_temp', 'nebulizer_flow',
        'collision_energy', 'dilution_factor'
    ]
    data_canonical = df[set1_vars + set2_vars].dropna()
    X = data_canonical[set1_vars].values
    Y = data_canonical[set2_vars].values

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    try:
        cancorr = CanCorr(Y_scaled, X_scaled)
        cancorr_results = cancorr.fit()
        print("Canonical Correlations:")
        for i, corr in enumerate(cancorr_results.cancorr):
            print(f"Canonical Pair {i+1}: {corr:.4f}")

        print("\nCanonical Loadings - Set 1 (Technique Properties):")
        for i, var in enumerate(set1_vars):
            print(f"{var}: {cancorr_results.x_cancoef[i, 0]:.4f}")

        print("\nCanonical Loadings - Set 2 (Instrumental Parameters):")
        for i, var in enumerate(set2_vars):
            print(f"{var}: {cancorr_results.y_cancoef[i, 0]:.4f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        canonical_scores_x = X_scaled @ cancorr_results.x_cancoef[:, 0]
        canonical_scores_y = Y_scaled @ cancorr_results.y_cancoef[:, 0]
        ax1.scatter(canonical_scores_x, canonical_scores_y, alpha=0.6)
        ax1.set_xlabel(f'Canonical Scores X (r = {cancorr_results.cancorr[0]:.3f})')
        ax1.set_ylabel('Canonical Scores Y')
        ax1.set_title('First Pair of Canonical Variables')

        feature_names = set1_vars + set2_vars
        all_loadings = np.vstack([
            cancorr_results.x_cancoef[:, 0],
            cancorr_results.y_cancoef[:, 0]
        ])
        for i, name in enumerate(feature_names):
            loading = all_loadings[i, 0] if i < len(set1_vars) else all_loadings[i, 0]
            ax2.arrow(0, 0, loading * 3, 0, head_width=0.1, head_length=0.1,
                      fc='red' if i < len(set1_vars) else 'blue',
                      ec='red' if i < len(set1_vars) else 'blue')
            ax2.text(loading * 3.2, 0.1, name, fontsize=10, rotation=45)
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-0.5, 1)
        ax2.set_title('Canonical Variable Loadings')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in canonical analysis: {e}")
        print("Performing alternative PCA...")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(np.hstack([X_scaled, Y_scaled]))
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} of variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} of variance)')
        plt.title('Alternative Principal Component Analysis')
        plt.show()


def discriminant_analysis(df):
    """
    Performs Linear Discriminant Analysis (LDA) to classify extraction techniques.

    This function uses experimental parameters and results as features to
    build a classifier that can distinguish between the different extraction
    techniques. It reports the classification accuracy and visualizes the
    separation of the techniques in the discriminant space.

    Args:
        df (pd.DataFrame): The input dataframe.
    """
    print("\n=== Discriminant Analysis ===\n")
    feature_cols = [
        'spray_voltage', 'capillary_temp', 'nebulizer_flow',
        'collision_energy', 'matrix_effect', 'recovery',
        'is_response_ratio'
    ]
    X = df[feature_cols].dropna()
    y = df.loc[X.index, 'technique']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    lda = LDA()
    lda.fit(X_train, y_train)
    X_lda = lda.transform(X_scaled)

    plt.figure(figsize=(12, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, technique in enumerate(df['technique'].unique()):
        mask = y == technique
        plt.scatter(
            X_lda[mask, 0], X_lda[mask, 1],
            c=colors[i], label=technique, alpha=0.7
        )
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title('Discriminant Analysis - Extraction Techniques')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    train_score = lda.score(X_train, y_train)
    test_score = lda.score(X_test, y_test)
    print(f"Accuracy on training set: {train_score:.3f}")
    print(f"Accuracy on test set: {test_score:.3f}")

    print("\nVariable Contributions (LDA coefficients):")
    for i, var in enumerate(feature_cols):
        print(f"{var}: {lda.coef_[0, i]:.4f}")
