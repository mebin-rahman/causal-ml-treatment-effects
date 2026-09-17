"""
03 - Causal DAG + Refutation Tests

A CATE number from script 02 is only trustworthy if the causal assumptions
behind it are made explicit and stress-tested. 

Steps:
1. Encode the assumed causal DAG (which variables confound, which don't).
2. Ask DoWhy to identify the estimand under that DAG (backdoor adjustment).
3. Estimate the effect.
4. Refute it with two standard sensitivity checks:
   - Placebo treatment: replace real treatment with random noise -> effect
     should collapse to ~0. If it doesn't, something is leaking/wrong.
   - Random common cause: add an irrelevant random covariate -> estimate
     should barely move. If it does, the model is unstable.
"""
import numpy as np
import pandas as pd
from causaldata import nhefs
from dowhy import CausalModel
from sklearn.impute import SimpleImputer

RNG = 42
np.random.seed(RNG)

CONFOUNDERS = [
    "age", "sex", "race", "education", "smokeintensity", "smokeyrs",
    "exercise", "active", "wt71", "ht",
]
TREATMENT = "qsmk"
OUTCOME = "wt82_71"

GRAPH = "digraph {"
for c in CONFOUNDERS:
    GRAPH += f'"{c}" -> "{TREATMENT}"; "{c}" -> "{OUTCOME}"; '
GRAPH += f'"{TREATMENT}" -> "{OUTCOME}";'
GRAPH += "}"


def load_data():
    df = nhefs.load_pandas().data
    cols = CONFOUNDERS + [TREATMENT, OUTCOME]
    df = df[cols].dropna(subset=[TREATMENT, OUTCOME]).copy()
    imputer = SimpleImputer(strategy="median")
    df[CONFOUNDERS] = imputer.fit_transform(df[CONFOUNDERS])
    return df


if __name__ == "__main__":
    df = load_data()

    model = CausalModel(
        data=df,
        treatment=TREATMENT,
        outcome=OUTCOME,
        graph=GRAPH,
    )
    model.view_model(layout="dot")
    print("Saved DAG visualization to causal_model.png")

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print("=== Identified estimand ===")
    print(identified_estimand)

    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
    )
    print("\n=== Estimate (linear backdoor adjustment) ===")
    print(f"ATE = {estimate.value:+.2f} kg")

    print("\n=== Refutation 1: placebo treatment ===")
    refute_placebo = model.refute_estimate(
        identified_estimand, estimate,
        method_name="placebo_treatment_refuter",
        placebo_type="permute",
        random_state=RNG,
    )
    print(refute_placebo)

    print("\n=== Refutation 2: random common cause ===")
    refute_random_cause = model.refute_estimate(
        identified_estimand, estimate,
        method_name="random_common_cause",
        random_state=RNG,
    )
    print(refute_random_cause)

    print("\nInterpretation: if the placebo effect is ~0 and the random-"
          "common-cause estimate barely moves from the original, that's "
          "evidence the ATE isn't an artifact of the specific model/data "
          "split -- it survives basic stress tests. Report both refutation "
          "results alongside the point estimate; a number without a "
          "sensitivity check is not a trustworthy claim.")
