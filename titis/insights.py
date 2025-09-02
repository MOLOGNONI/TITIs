import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def generate_insights(df, correlation_matrix, significant_correlations):
    """
    Generates automatic insights based on the statistical analyses.
    """
    print("\n=== INSIGHTS E INTERPRETAÇÃO AUTOMÁTICA ===\n")

    print("1. PADRÕES DE CORRELAÇÃO DESCOBERTOS:")
    print("-" * 40)

    # Analysis of significant correlations
    strong_correlations = [corr for corr in significant_correlations if abs(corr['correlation']) > 0.7]
    if strong_correlations:
        print("Correlações Fortes (|r| > 0.7):")
        for corr in strong_correlations:
            direction = "positiva" if corr['correlation'] > 0 else "negativa"
            print(f"• {corr['var1']} ↔ {corr['var2']}: {direction} ({corr['correlation']:.3f})")

    print(f"\n2. ANÁLISE POR TÉCNICA DE EXTRAÇÃO:")
    print("-" * 40)

    # Statistics by technique
    for technique in df['technique'].unique():
        subset = df[df['technique'] == technique]
        print(f"\n{technique}:")
        print(f"  Matrix Effect médio: {subset['matrix_effect'].mean():.1f}% ± {subset['matrix_effect'].std():.1f}")
        print(f"  Recovery médio: {subset['recovery'].mean():.3f} ± {subset['recovery'].std():.3f}")
        print(f"  IS Response Ratio: {subset['is_response_ratio'].mean():.3f} ± {subset['is_response_ratio'].std():.3f}")

    print(f"\n3. EFEITOS DA DILUIÇÃO:")
    print("-" * 40)

    # Analysis by dilution factor
    print("Fator de Diluição vs Matrix Effect:")
    for dilution, group in df.groupby('dilution_factor'):
        print(f"  {dilution}x: ME = {group['matrix_effect'].mean():.1f}%, Recovery = {group['recovery'].mean():.3f}")

    print(f"\n4. VARIÁVEIS MAIS INFLUENTES:")
    print("-" * 40)

    # Random Forest for feature importance
    feature_cols = ['spray_voltage', 'capillary_temp', 'nebulizer_flow',
                   'collision_energy', 'cleanup_power', 'specificity']
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df['matrix_effect'].fillna(df['matrix_effect'].mean())

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    feature_importance = list(zip(feature_cols, rf.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print("Importância para predição do Matrix Effect:")
    for feature, importance in feature_importance:
        print(f"  {feature}: {importance:.3f}")

    print(f"\n5. RECOMENDAÇÕES OPERACIONAIS:")
    print("-" * 40)

    # Technique with best performance
    best_technique = df.groupby('technique').agg({
        'matrix_effect': lambda x: abs(x).mean(),  # Menor efeito matriz absoluto
        'recovery': 'mean',
        'is_response_ratio': lambda x: abs(x - 1).mean()  # Mais próximo de 1
    })

    # Composite score (lower is better)
    best_technique['score'] = (best_technique['matrix_effect'] +
                              (1 - best_technique['recovery']) * 100 +
                              best_technique['is_response_ratio'] * 100)
    best_tech = best_technique['score'].idxmin()

    print(f"• Técnica recomendada: {best_tech}")
    print(f"  (Menor efeito matriz, melhor recovery, IS response mais estável)")

    # Optimal instrumental parameters
    optimal_voltage = df.loc[df['matrix_effect'].abs().idxmin(), 'spray_voltage']
    optimal_temp = df.loc[df['matrix_effect'].abs().idxmin(), 'capillary_temp']

    print(f"• Parâmetros instrumentais sugeridos:")
    print(f"  Spray Voltage: ~{optimal_voltage:.0f}V")
    print(f"  Capillary Temperature: ~{optimal_temp:.0f}°C")

    # Optimal dilution
    optimal_dilution = df.groupby('dilution_factor').agg({
        'matrix_effect': lambda x: abs(x).mean()
    })['matrix_effect'].idxmin()

    print(f"• Fator de diluição recomendado: {optimal_dilution}x")
    print(f"  (Minimiza efeito matriz mantendo sensibilidade)")
