###################################################
# PyMC: Bayesian Hierarchical Logistic + L1 Prior
# Loop over multiple genes to produce:
#   - A forest plot of climate effects for each gene
#   - Partial dependence plots (all climate variables) in a multi-panel figure for each gene
#   - A combined summary CSV of model estimates (for Supplementary Material)
# Input Files: gene_presence_data.csv, average_climate_data.csv
###################################################

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import sys

#############################################
# 1. READ & MERGE DATA
#############################################

df_genes = pd.read_csv("gene_presence_data.csv")
df_climate = pd.read_csv("average_climate_data.csv")

# Clean column names: remove leading/trailing spaces
df_genes.columns = df_genes.columns.str.strip()
df_climate.columns = df_climate.columns.str.strip()

# Rename 'Date' to 'quarter_year' 
if "quarter_year" not in df_genes.columns:
    if "Date" in df_genes.columns:
        df_genes.rename(columns={"Date": "quarter_year"}, inplace=True)
    else:
        print("Error: gene dataset must have 'quarter_year' or 'Date' column.")
        sys.exit(1)
if "quarter_year" not in df_climate.columns:
    if "Date" in df_climate.columns:
        df_climate.rename(columns={"Date": "quarter_year"}, inplace=True)
    else:
        print("Error: climate dataset must have 'quarter_year' or 'Date' column.")
        sys.exit(1)

# Merge using inner join
df_merged = pd.merge(df_genes, df_climate, on="quarter_year", how="inner")
print("Shape of df_merged:", df_merged.shape)

#############################################
# 2. DEFINE CLIMATE VARIABLES & GENES
#############################################

# List all 11 climate variables as they appear in df_climate
climate_vars = ["AWND", "DP01", "DP10", "DX70", "DX90", "EMXT", "PRCP", "TAVG", "TMAX", "TMIN", "RUNOFF"]

# Define the genes to analyze
genes_of_interest = [
    "ADE17", "ADE17_G45V", "CAT1", "CDR1", "CDR1_V704L", "CDR2", "ERG11", "ERG11_I466M",
    "ERG11_Y132F", "ERG3", "FCY2", "FKS1", "FKS1_D642Y", "FKS1_F635C", "FKS1_S638Y",
    "FKS1_S639F", "FKS1_S639P", "FKS1_S639Y", "FKS2", "FLO8", "FUR1", "FUR1_F211I",
    "FUR1_Q64", "HSP104", "HSP90", "MRR1", "MRR1_N647T", "TAC1a", "TAC1b", "TAC1b_A640V",
    "TAC1b_A657V", "TAC1b_F214L", "TAC1b_K247E"
]

# Combined summary for all models
combined_summary = pd.DataFrame()

#############################################
# 4. DEFINE PARTIAL DEPENDENCE FUNCTION
#############################################
def partial_dependence_all(df_base, idata, climate_vars, grid_points=50, quarter_effect=0.0):
    """
    Compute approximate partial dependence for each climate variable.
    Fix other climate variables at their median value from df_base.
    Returns a dictionary: {var_name: (grid, predictions)}.
    """
    # Get posterior means using InferenceData style
    intercept_mean = idata.posterior["intercept"].mean(dim=["chain", "draw"]).item()
    beta_means = idata.posterior["beta"].mean(dim=["chain", "draw"]).values  # shape (n_features,)
    
    results = {}
    for var in climate_vars:
        try:
            idx = climate_vars.index(var)
        except ValueError:
            continue
        # Create a grid for the current variable
        var_min = df_base[var].min()
        var_max = df_base[var].max()
        grid = np.linspace(var_min, var_max, grid_points)
        
        # Fix all other variables at their median values
        base_vals = np.array([df_base[cv].median() for cv in climate_vars])
        preds = []
        for val in grid:
            new_vals = base_vals.copy()
            new_vals[idx] = val
            lin = intercept_mean + np.dot(new_vals, beta_means) + quarter_effect
            p_val = 1.0 / (1.0 + np.exp(-lin))
            preds.append(p_val)
        results[var] = (grid, preds)
    return results

#############################################
# 5. LOOP OVER GENES & FIT PYMC MODEL
#############################################

for gene in genes_of_interest:
    print(f"\n=== Processing gene: {gene} ===")
    
    # Check if the gene exists in the merged DataFrame
    if gene not in df_merged.columns:
        print(f"Warning: {gene} not found in merged data. Skipping.")
        continue
    
    # Set outcome for current gene
    df_merged["gene_outcome"] = df_merged[gene]
    
    # Drop rows with NA in gene outcome or any climate variable
    df_sub = df_merged.dropna(subset=["gene_outcome"] + climate_vars).copy()
    df_sub["quarter_year"] = df_sub["quarter_year"].astype("category")
    
    # Check variation
    zeros = sum(df_sub["gene_outcome"] == 0)
    ones  = sum(df_sub["gene_outcome"] == 1)
    print(f"Gene {gene} -> zeros: {zeros}, ones: {ones}")
    if zeros < 1 or ones < 1:
        print(f"No variation in {gene}. Skipping.")
        continue
    
    # Build data arrays for PyMC
    quarter_idx = df_sub["quarter_year"].cat.codes.values
    n_quarters = df_sub["quarter_year"].nunique()
    X = df_sub[climate_vars].values
    n_features = len(climate_vars)
    y = df_sub["gene_outcome"].values
    
    # Fit the PyMC model
    with pm.Model() as model:
        sigma_q = pm.HalfNormal("sigma_q", sigma=1.0)
        quarter_effect = pm.Normal("quarter_effect", mu=0, sigma=sigma_q, shape=n_quarters)
        beta = pm.Laplace("beta", mu=0.0, b=1.0, shape=n_features)
        intercept = pm.Normal("intercept", mu=0, sigma=2.0)
        mu_lin = intercept + pm.math.dot(X, beta) + quarter_effect[quarter_idx]
        p = pm.invlogit(mu_lin)
        y_obs = pm.Bernoulli("y_obs", p=p, observed=y)
        
        trace = pm.sample(draws=2000, tune=1000, chains=4, cores=4, target_accept=0.9, progressbar=False)
        
        # Summarize model parameters and add gene label
        summary_main = az.summary(trace, var_names=["intercept", "beta", "sigma_q"], round_to=4)
        summary_main["gene"] = gene
        combined_summary = pd.concat([combined_summary, summary_main], ignore_index=True)
    
    # Save individual model summary
    summary_main.to_csv(f"model_summary_{gene}.csv", index=False)
    print(f"Saved model summary for {gene} to model_summary_{gene}.csv")
    
    #############################################
    # VISUALIZATION: FOREST PLOT
    #############################################
    beta_summary = az.summary(trace, var_names=["beta"], round_to=4)
    beta_summary["Variable"] = climate_vars
    beta_summary.sort_values("mean", inplace=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        beta_summary["mean"],
        beta_summary["Variable"],
        xerr=[
            beta_summary["mean"] - beta_summary["hdi_3%"],
            beta_summary["hdi_97%"] - beta_summary["mean"]
        ],
        fmt="o",
        color="darkblue",
        ecolor="gray",
        capsize=4,
        markersize=6
    )
    ax.axvline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("Posterior Mean (Log-Odds Effect)", fontsize=12)
    ax.set_ylabel("Climate Variable", fontsize=12)
    ax.set_title(f"Forest Plot of Climate Effects on Gene Presence ({gene})", fontsize=14)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    forest_filename = f"climate_forest_plot_{gene}.png"
    plt.savefig(forest_filename, dpi=300)
    plt.close(fig)
    print(f"Saved forest plot for {gene} to {forest_filename}")
    
    #############################################
    # VISUALIZATION: PARTIAL DEPENDENCE PLOTS FOR ALL CLIMATE VARIABLES
    #############################################
    pd_results = partial_dependence_all(df_sub, trace, climate_vars, grid_points=50, quarter_effect=0.0)
    
    # Create subplots: Let's use 3 rows x 4 columns (for 11 variables, one subplot may be blank)
    n_vars = len(climate_vars)
    n_cols = 4
    n_rows = int(np.ceil(n_vars / n_cols))
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*2.5), squeeze=False)
    axs = axs.flatten()
    
    for i, var in enumerate(climate_vars):
        grid, preds = pd_results[var]
        axs[i].plot(grid, preds, color="darkgreen", lw=2)
        axs[i].set_title(var, fontsize=10)
        axs[i].set_xlabel(var, fontsize=8)
        axs[i].set_ylabel("Probability", fontsize=8)
        axs[i].grid(True, linestyle="--", alpha=0.6)
    # Hide any unused subplots
    for j in range(n_vars, len(axs)):
        axs[j].axis("off")
        
    plt.suptitle(f"Partial Dependence Plots for {gene}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pd_filename = f"partial_dependence_all_{gene}.png"
    plt.savefig(pd_filename, dpi=300)
    plt.close(fig)
    print(f"Saved partial dependence plots for {gene} to {pd_filename}")

#############################################
# 6. EXPORT COMBINED MODEL SUMMARY
#############################################
combined_summary.to_csv("combined_model_summary.csv", index=False)
print("\nCombined model summary for all genes saved to 'combined_model_summary.csv'.")
print("\nAll done with PyMC model and visualizations for all genes.\n")
