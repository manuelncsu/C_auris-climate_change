#!/usr/bin/env python3
# ------------------------------------------------------------
# Candida auris climate-gene Bayesian analysis
#  – full parameter CSV +
#    OR forest plots with 2-digit tick labels
#    + trace plots (Fig S2) & prior–post overlays (Fig S3)
# ------------------------------------------------------------
import sys, warnings
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import laplace

# ── User tunables ──────────────────────────────────────────
TARGET_ACCEPT = 0.95
DRAWS, TUNE   = 2000, 1000
CHAINS, CORES = 4, 4
HDI_PROB      = 0.94

# ── 1.  Data load & merge ─────────────────────────────────
GENE_FILE, CLIMATE_FILE = "gene_presence_data.csv", "average_climate_data.csv"
df_genes   = pd.read_csv(GENE_FILE)
df_climate = pd.read_csv(CLIMATE_FILE)

def ensure_quarter(df, cols=("quarter_year","Date","date","collection_quarter")):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df.rename(columns={c:"quarter_year"}, inplace=True)
            break
    if pd.api.types.is_datetime64_any_dtype(df["quarter_year"]):
        df["quarter_year"] = (
            pd.to_datetime(df["quarter_year"])
              .dt.to_period("Q")
              .astype(str)
        )
    return df

df_genes   = ensure_quarter(df_genes)
df_climate = ensure_quarter(df_climate)
df_all     = pd.merge(df_genes, df_climate, on="quarter_year", how="inner")
print("✓ merged shape:", df_all.shape)

climate_vars = ["AWND","DP01","DP10","DX70","DX90","EMXT",
                "PRCP","TAVG","TMAX","TMIN","RUNOFF"]

genes = ["ADE17","ADE17_G45V","CAT1","CDR1","CDR1_V704L","CDR2","ERG11",
         "ERG11_I466M","ERG11_Y132F","ERG3","FCY2","FKS1","FKS1_D642Y",
         "FKS1_F635C","FKS1_S638Y","FKS1_S639F","FKS1_S639P","FKS1_S639Y",
         "FKS2","FLO8","FUR1","FUR1_F211I","FUR1_Q64","HSP104","HSP90",
         "MRR1","MRR1_N647T","TAC1a","TAC1b","TAC1b_A640V","TAC1b_A657V",
         "TAC1b_F214L","TAC1b_K247E"]

full_table = pd.DataFrame()

# ── helper for partial-dependence (unchanged) ─────────────
def partial_dependence_all(df, idata, vars_, pts=50, q_eff=0.0):
    b_mean = idata.posterior["beta"].mean(("chain","draw")).values
    a_mean = idata.posterior["intercept"].mean().item()
    out    = {}
    for i,var in enumerate(vars_):
        if var not in df.columns:
            continue
        lo,hi = df[var].quantile([0.05,0.95])
        grid  = np.linspace(lo,hi,pts)
        base  = np.array([df[c].median() for c in vars_])
        preds = []
        for x in grid:
            tmp   = base.copy()
            tmp[i] = x
            lin   = a_mean + np.dot(tmp, b_mean) + q_eff
            preds.append(1/(1+np.exp(-lin)))
        out[var] = (grid, preds)
    return out

# ── 2.  Model loop ──────────────────────────────────────────
for gene in genes:
    if gene not in df_all.columns:
        warnings.warn(f"{gene} absent – skipped.")
        continue

    print(f"\n=== {gene} ===")
    df_sub = (
        df_all[["quarter_year", gene] + climate_vars]
        .dropna()
        .rename(columns={gene: "y"})
    )
    if df_sub.empty:
        warnings.warn(f"No data for {gene} – skipped.")
        continue

    y        = df_sub["y"].astype(int).values
    X        = df_sub[climate_vars].values
    q_idx, _ = pd.factorize(df_sub["quarter_year"])
    n_q, n_f = len(np.unique(q_idx)), X.shape[1]

    with pm.Model() as model:
        σ_q   = pm.HalfNormal("sigma_q", 1.0)
        q_eff = pm.Normal("quarter_eff", 0.0, σ_q, shape=n_q)
        β     = pm.Laplace("beta", 0.0, 1.0, shape=n_f)
        α     = pm.Normal("intercept", 0.0, 2.0)

        lin   = α + pm.math.dot(X, β) + q_eff[q_idx]
        pm.Bernoulli("y_obs", logit_p=lin, observed=y)

        idata = pm.sample(
            draws        = DRAWS,
            tune         = TUNE,
            chains       = CHAINS,
            cores        = CORES,
            target_accept= TARGET_ACCEPT,
            return_inferencedata=True,
            progressbar=False,
        )

    # ── A. Odds‐ratio forest (Fig 2) ─────────────────────────
    beta_da = idata.posterior["beta"]
    feat_dim= beta_da.dims[-1]       # e.g. "beta_dim_0"
    or_da    = np.exp(beta_da).assign_coords({feat_dim: climate_vars})

    az.plot_forest(or_da, combined=True, hdi_prob=HDI_PROB)
    ax = plt.gca()
    ax.set_xscale("log")
    ax.axvline(1, ls="--", c="k")

    # dynamic half-decade ticks with ≤2-sig-fig labels
    xmin,xmax = or_da.quantile([0.005,0.995]).values
    dmin,dmax = np.floor(np.log10(xmin)), np.ceil(np.log10(xmax))
    ticks     = np.logspace(dmin, dmax, int((dmax-dmin)/0.5)+1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.2g}" for t in ticks])

    plt.title(f"{gene}: climate effects (OR)")
    plt.xlabel("Odds ratio")
    plt.tight_layout()
    plt.savefig(f"Fig2_OR_{gene}.png", dpi=300)
    plt.close()

    # ── B. Summarize scalars + betas in table ─────────────────
    sm = az.summary(
        idata,
        var_names   = ["intercept", "sigma_q", "beta"],
        hdi_prob    = HDI_PROB,
        round_to    = 4,
    ).reset_index().rename(columns={"index":"parameter"})

    # split out 'param' and 'covariate' columns
    def split_param(p):
        if isinstance(p, tuple):
            return "beta", climate_vars[p[1]]
        return p, np.nan

    sm[["param","covariate"]] = sm["parameter"].apply(split_param).tolist()
    sm.drop(columns="parameter", inplace=True)
    sm["gene"] = gene

    full_table = pd.concat([full_table, sm], ignore_index=True)

    # ── C. Fig S2: trace plots for intercept, σ_q, and significant β’s ──
    # find which betas are credibly non-zero
    sig_ix = []
    for i in range(n_f):
        post = idata.posterior["beta"][:,:,i].values.ravel()
        lo, hi = np.percentile(post, [(1-HDI_PROB)/2*100, (1+HDI_PROB)/2*100])
        if lo > 0 or hi < 0:
            sig_ix.append(i)

    # traces for intercept & sigma_q
    for var in ["intercept","sigma_q"]:
        az.plot_trace(idata, var_names=[var], compact=False)
        plt.tight_layout()
        plt.savefig(f"FigS2_traces_{gene}_{var}.png", dpi=300)
        plt.close()

    # traces for each significant climate coefficient
    for i in sig_ix:
        cov = climate_vars[i]
        az.plot_trace(
            idata,
            var_names=["beta"],
            coords={feat_dim:[i]},
            compact=False
        )
        plt.suptitle(f"{gene} – {cov}")
        plt.tight_layout()
        plt.savefig(f"FigS2_traces_{gene}_beta_{cov}.png", dpi=300)
        plt.close()

    # ── D. Fig S3: prior vs posterior overlays for those same β’s ───
    xg  = np.linspace(-4, 4, 400)
    lpdf= laplace.pdf(xg, 0, 1.0)
    for i in sig_ix:
        cov  = climate_vars[i]
        post = idata.posterior["beta"][:,:,i].values.ravel()

        plt.figure(figsize=(4,3))
        plt.hist(post, bins=30, density=True, alpha=0.5, label="Posterior")
        plt.plot(xg, lpdf, lw=2, label="Prior (Laplace)")
        plt.axvline(0, ls="--", c="k")
        plt.title(f"{gene} – {cov}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"FigS3_prior_post_{gene}_{cov}.png", dpi=300)
        plt.close()

# ── 3.  Save combined summary ───────────────────────────────
full_table.to_csv("combined_model_summary.csv", index=False)
print("\n✔  Finished. CSV → combined_model_summary.csv\n")
