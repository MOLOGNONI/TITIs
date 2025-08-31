import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from titis.data.generation import generate_titis_experimental_data
from titis.analysis.statistics import (
    set_professional_style,
    analyze_variable_dependencies,
    perform_manova_analysis,
    canonical_correlation_analysis,
    discriminant_analysis
)
from titis.visualization.plotting import (
    advanced_visualizations,
    generate_insights
)


def main():
    """
    Main function to run the full TITIs statistical analysis workflow.
    """
    print("=" * 60)
    print("TITIs ADVANCED STATISTICAL ANALYSIS")
    print("=" * 60)

    # Set the plot style
    set_professional_style()

    # 1. Generate data
    print("\nGenerating experimental data...")
    df, molecular_df = generate_titis_experimental_data()
    print(f"Dataset created: {len(df)} experiments, {len(molecular_df)} compounds")
    print(f"Techniques analyzed: {df['technique'].unique().tolist()}")

    # 2. Perform sequential analyses
    correlation_matrix, significant_correlations = analyze_variable_dependencies(df)
    perform_manova_analysis(df)
    canonical_correlation_analysis(df)
    discriminant_analysis(df)

    # 3. Create visualizations
    advanced_visualizations(df)

    # 4. Generate insights
    generate_insights(df, correlation_matrix, significant_correlations)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("The data has revealed its fundamental patterns!")
    print("=" * 60)


if __name__ == "__main__":
    main()
