"""
02 - NHEFS: Real-World CATE Estimation
========================================
Dataset: NHEFS (NHANES Epidemiologic Follow-up Study), the dataset used
throughout Hernan & Robins, "Causal Inference: What If" 

Question: does quitting smoking (treatment, `qsmk`) causally affect weight
gain (outcome, `wt82_71`, in kg between 1971 and 1982)? This is a classic
"treatment decision -> downstream outcome with real confounding" problem.

Confounders: age, sex, race, education, baseline weight, smoking intensity/
years, exercise, physical activity -- healthier/heavier smokers are more
likely to quit AND more likely to gain weight regardless, so a naive
correlation between quitting and weight gain is confounded.

We go beyond the single ATE and ask: does the effect of quitting
vary by baseline smoking intensity or age (CATE)? For example, who benefits most/
is most at risk, since treatment effects vary across a population.
"""
import numpy as np
import pandas as pd
from causaldata import nhefs
from econml.metalearners import TLearner, XLearner
from econml.grf import CausalForest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer

RNG = 42
np.random.seed(RNG)

CONFOUNDERS = [
    "age", "sex", "race", "education", "smokeintensity", "smokeyrs",
    "exercise", "active", "wt71", "ht",
]
TREATMENT = "qsmk"       # 1 = quit smoking between baseline and follow-up
OUTCOME = "wt82_71"      # weight change in kg, 1971 -> 1982
EFFECT_MODIFIERS = ["smokeintensity", "age"]  # for CATE heterogeneity


def load_data():
    df = nhefs.load_pandas().data
    cols = CONFOUNDERS + [TREATMENT, OUTCOME]
    df = df[cols].dropna(subset=[TREATMENT, OUTCOME]).copy()
    # simple median imputation for the few missing confounder values
    imputer = SimpleImputer(strategy="median")
    df[CONFOUNDERS] = imputer.fit_transform(df[CONFOUNDERS])
    return df


def naive_association(df):
    """What a purely correlational model would report -- the wrong
    number, kept here deliberately to show WHY causal ML is needed."""
    quit_mean = df.loc[df[TREATMENT] == 1, OUTCOME].mean()
    stay_mean = df.loc[df[TREATMENT] == 0, OUTCOME].mean()
    return quit_mean - stay_mean


def estimate_cate(df):
    X = df[CONFOUNDERS].values
    T = df[TREATMENT].values
    Y = df[OUTCOME].values

    x_learner = XLearner(models=GradientBoostingRegressor(random_state=RNG))
    x_learner.fit(Y, T, X=X)
    tau_hat = x_learner.effect(X)

    cf = CausalForest(n_estimators=1000, random_state=RNG, honest=True)
    cf.fit(X, T, Y)
    tau_hat_cf = cf.predict(X).flatten()
    # 90% confidence interval per unit, not just a point estimate
    
    lb, ub = cf.predict_interval(X, alpha=0.1)

    out = df.copy()
    out["tau_hat_xlearner"] = tau_hat
    out["tau_hat_causalforest"] = tau_hat_cf
    out["tau_hat_cf_lower90"] = lb.flatten()
    out["tau_hat_cf_upper90"] = ub.flatten()
    return out, x_learner, cf


def summarize_heterogeneity(out):
    print("\n--- CATE by baseline smoking intensity tercile ---")
    out["smoke_tercile"] = pd.qcut(out["smokeintensity"], 3,
                                    labels=["low", "mid", "high"])
    print(out.groupby("smoke_tercile")["tau_hat_causalforest"].mean())

    print("\n--- CATE by age group ---")
    out["age_group"] = pd.cut(out["age"], bins=[0, 40, 55, 100],
                               labels=["<=40", "41-55", "56+"])
    print(out.groupby("age_group")["tau_hat_causalforest"].mean())


if __name__ == "__main__":
    df = load_data()
    print(f"N = {len(df)} | Treated (quit smoking) = {df[TREATMENT].sum()} "
          f"({df[TREATMENT].mean():.1%})")

    naive = naive_association(df)
    print(f"\nNaive association (quit vs stay, no adjustment): "
          f"{naive:+.2f} kg")

    out, x_learner, cf = estimate_cate(df)
    ate_x = out["tau_hat_xlearner"].mean()
    ate_cf = out["tau_hat_causalforest"].mean()
    print(f"X-learner ATE (confounder-adjusted):   {ate_x:+.2f} kg")
    print(f"Causal Forest ATE (confounder-adjusted): {ate_cf:+.2f} kg")
    print(f"(Naive vs adjusted gap of "
          f"{abs(naive - ate_cf):.2f} kg is the confounding bias removed "
          f"by conditioning on {', '.join(CONFOUNDERS)})")

    summarize_heterogeneity(out)

    out.to_csv("results_nhefs_cate.csv", index=False)
    print("\nSaved: results_nhefs_cate.csv")
