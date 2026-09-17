"""
01 - IHDP Benchmark
====================
Goal: validate that we can correctly estimate heterogeneous treatment effects
(CATE) when we KNOW the ground truth, before trusting the same pipeline on
real data where we don't.

IHDP (Infant Health and Development Program) is the standard benchmark in
causal ML papers (Hill 2011, Shalit et al. 2017, Wager & Athey 2018, etc.)
because it's a semi-synthetic dataset: real covariates from a real RCT, but
simulated outcomes with a KNOWN individual treatment effect (tau). This lets
us score estimators on PEHE (Precision in Estimation of Heterogeneous Effect).


"""
import numpy as np
import pandas as pd
from econml.data.dgps import ihdp_surface_B
from econml.metalearners import TLearner, XLearner
from econml.grf import CausalForest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

RNG = 42
np.random.seed(RNG)


def pehe(tau_true, tau_hat):
    """Precision in Estimation of Heterogeneous Effect: RMSE between
    estimated and true individual treatment effects. Lower is better.
    This is THE standard metric in the causal ML literature because
    accuracy on Y doesn't tell you accuracy on tau -- a model can predict
    outcomes well while getting the treatment effect completely wrong."""
    return np.sqrt(np.mean((tau_true - tau_hat) ** 2))


def ate_error(tau_true, tau_hat):
    """Error in the single population-level Average Treatment Effect."""
    return abs(tau_true.mean() - tau_hat.mean())


def run_benchmark(n_repeats=10):
    results = []

    for rep in range(n_repeats):
        Y, T, X, tau_true = ihdp_surface_B()
        X_tr, X_te, T_tr, T_te, Y_tr, Y_te, tau_tr, tau_te = train_test_split(
            X, T, Y, tau_true, test_size=0.3, random_state=RNG + rep
        )

        # --- T-learner: separate outcome model per treatment arm ---
        t_learner = TLearner(models=GradientBoostingRegressor(random_state=RNG))
        t_learner.fit(Y_tr, T_tr, X=X_tr)
        tau_hat_t = t_learner.effect(X_te)

        # --- X-learner: T-learner + reweights by propensity, better with
        # imbalanced treatment groups (IHDP is ~19% treated) ---
        x_learner = XLearner(models=GradientBoostingRegressor(random_state=RNG))
        x_learner.fit(Y_tr, T_tr, X=X_tr)
        tau_hat_x = x_learner.effect(X_te)

        # --- Causal Forest: honest splitting, gives valid confidence
        
        cf = CausalForest(n_estimators=500, random_state=RNG, honest=True)
        cf.fit(X_tr, T_tr, Y_tr)
        tau_hat_cf = cf.predict(X_te).flatten()

        for name, tau_hat in [
            ("T-learner", tau_hat_t),
            ("X-learner", tau_hat_x),
            ("Causal Forest", tau_hat_cf),
        ]:
            results.append({
                "repeat": rep,
                "method": name,
                "PEHE": pehe(tau_te, tau_hat),
                "ATE_error": ate_error(tau_te, tau_hat),
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_benchmark(n_repeats=10)
    summary = df.groupby("method")[["PEHE", "ATE_error"]].agg(["mean", "std"])
    print("\n=== IHDP Benchmark: 10 repeats, 70/30 train/test split ===\n")
    print(summary)
    summary.to_csv("results_ihdp_benchmark.csv")
    df.to_csv("results_ihdp_benchmark_raw.csv", index=False)
    print("\nSaved: results_ihdp_benchmark.csv, results_ihdp_benchmark_raw.csv")