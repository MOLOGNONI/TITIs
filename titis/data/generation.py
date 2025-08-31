import numpy as np
import pandas as pd


def generate_titis_experimental_data():
    """
    Generates a realistic, simulated dataset for TITIs analysis.

    This function creates a pandas DataFrame of experimental results and another
    DataFrame of molecular properties. The data is designed to mimic real-world
    LC-MS/MS experiments, incorporating variability from extraction techniques,
    instrumental parameters, and the physicochemical properties of different
    compounds.

    The simulation is based on a complex, multi-factorial model to ensure
    the data is suitable for advanced statistical analysis like MANOVA and
    Canonical Correlation.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - experimental_df (pd.DataFrame): DataFrame with simulated
              experimental runs. Columns include 'technique', 'dilution_factor',
              'matrix_effect', 'recovery', etc.
            - molecular_properties_df (pd.DataFrame): DataFrame with the
              physicochemical properties of the simulated compounds.
    """
    np.random.seed(42)
    n_compounds = 100
    n_experiments = 500

    # Extraction techniques with distinct characteristics
    extraction_techniques = {
        'SLE-LTP': {'cleanup': 0.3, 'specificity': 0.4, 'recovery': 0.85, 'variability': 0.15},
        'QuEChERS': {'cleanup': 0.6, 'specificity': 0.7, 'recovery': 0.90, 'variability': 0.10},
        'SPE-OASIS': {'cleanup': 0.8, 'specificity': 0.6, 'recovery': 0.95, 'variability': 0.08},
        'SPE-MIP': {'cleanup': 0.9, 'specificity': 0.9, 'recovery': 0.88, 'variability': 0.05},
        'PLE-EDGE': {'cleanup': 0.7, 'specificity': 0.8, 'recovery': 0.92, 'variability': 0.12}
    }

    # Simulated molecular properties
    molecular_properties = pd.DataFrame({
        'compound_id': range(n_compounds),
        'log_p': np.random.normal(2.5, 1.5, n_compounds),
        'mol_weight': np.random.normal(300, 100, n_compounds),
        'pka': np.random.normal(7.5, 2.0, n_compounds),
        'polar_surface_area': np.random.exponential(80, n_compounds),
        'hbond_donors': np.random.poisson(2, n_compounds),
        'hbond_acceptors': np.random.poisson(4, n_compounds),
        'rotatable_bonds': np.random.poisson(6, n_compounds)
    })

    # Experimental data
    experimental_data = []

    for i in range(n_experiments):
        technique = np.random.choice(list(extraction_techniques.keys()))
        compound_idx = np.random.randint(0, n_compounds)

        # Instrumental parameters
        spray_voltage = np.random.normal(4000, 500)
        capillary_temp = np.random.normal(300, 50)
        nebulizer_flow = np.random.normal(40, 10)
        collision_energy = np.random.normal(25, 10)

        # Dilution factors
        dilution_factor = np.random.choice([1, 10, 100, 1000, 10000])

        # Calculation of matrix effect based on multiple factors
        tech_params = extraction_techniques[technique]
        mol_props = molecular_properties.iloc[compound_idx]

        # Complex model for matrix effect
        base_me = -50 + tech_params['cleanup'] * 40
        logp_effect = mol_props['log_p'] * 5
        dilution_effect = np.log10(dilution_factor) * 10

        # Instrumental factors
        voltage_effect = (spray_voltage - 4000) / 1000 * 5
        temp_effect = (capillary_temp - 300) / 100 * 3

        # Final matrix effect
        matrix_effect = (
            base_me + logp_effect + dilution_effect +
            voltage_effect + temp_effect +
            np.random.normal(0, tech_params['variability'] * 20)
        )

        # Recovery based on properties
        recovery = (
            tech_params['recovery'] -
            abs(mol_props['log_p'] - 2.5) * 0.05 +
            np.random.normal(0, 0.05)
        )
        recovery = np.clip(recovery, 0.5, 1.0)

        # Internal standard response
        is_response_ratio = np.random.normal(0.8, 0.15)

        experimental_data.append({
            'experiment_id': i,
            'technique': technique,
            'compound_id': compound_idx,
            'spray_voltage': spray_voltage,
            'capillary_temp': capillary_temp,
            'nebulizer_flow': nebulizer_flow,
            'collision_energy': collision_energy,
            'dilution_factor': dilution_factor,
            'matrix_effect': matrix_effect,
            'recovery': recovery,
            'is_response_ratio': is_response_ratio,
            'cleanup_power': tech_params['cleanup'],
            'specificity': tech_params['specificity'],
            'base_recovery': tech_params['recovery'],
            'variability': tech_params['variability']
        })

    return pd.DataFrame(experimental_data), molecular_properties
