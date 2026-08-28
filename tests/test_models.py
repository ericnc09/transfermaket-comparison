"""Gates on the model layer.

These are contract tests, not accuracy tests: accuracy moves with the data, but
a model that returns the wrong units or an inverted interval is broken whatever
the leaderboard says.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.eval.splits import rolling_origin  # noqa: E402
from src.models import baselines as B  # noqa: E402
from src.models.ensemble import _to_target_space  # noqa: E402
from src.models.gbm import TARGET, LightGBM, _reinflate  # noqa: E402
from src.models.quantile import Conformal, QuantileTrio, interval_report  # noqa: E402


@pytest.fixture(scope="module")
def fold():
    df = pd.read_parquet(ROOT / "data/processed/panel_model.parquet")
    folds = list(rolling_origin(df))
    # Use the largest fold so the test is stable as seasons are added.
    return max(folds, key=lambda f: len(f[2]))[1:]


def test_rolling_origin_never_trains_on_the_future():
    df = pd.read_parquet(ROOT / "data/processed/panel_model.parquet")
    for season, train, test in rolling_origin(df):
        assert train.Season_End_Year.max() < season
        assert set(test.Season_End_Year) == {season}


def test_reinflate_inverts_the_deflation(fold):
    """Predicting the target exactly must reproduce the euro value exactly."""
    train, _ = fold
    exact = _reinflate(train[TARGET].to_numpy(), train)
    assert np.allclose(exact, train.value_eur.to_numpy(), rtol=1e-6)


def test_stacker_target_space_roundtrip(fold):
    """Euro -> target space must be the inverse of re-inflation."""
    train, _ = fold
    back = _reinflate(_to_target_space(train.value_eur.to_numpy(), train), train)
    assert np.allclose(back, train.value_eur.to_numpy(), rtol=1e-6)


def test_baselines_predict_plausible_euros(fold):
    train, test = fold
    for cls in B.ALL:
        p = cls().fit(train).predict(test)
        assert len(p) == len(test)
        assert np.isfinite(p).all(), f"{cls.name} produced non-finite predictions"
        assert (p > 1e4).all() and (p < 5e8).all(), f"{cls.name} out of plausible range"


def test_quantile_intervals_are_ordered(fold):
    train, test = fold
    iv = QuantileTrio("coldstart").fit(train).predict_interval(test)
    assert (iv.p10 <= iv.p50).all()
    assert (iv.p50 <= iv.p90).all()


def test_conformal_coverage_is_near_nominal(fold):
    """Split conformal should land close to its nominal level by construction."""
    train, test = fold
    c = Conformal(LightGBM("coldstart", n_estimators=300), confidence=0.8).fit(train)
    picp = interval_report(c.predict_interval(test), test.value_eur.to_numpy())["picp"]
    assert 0.70 <= picp <= 0.90, f"conformal coverage {picp:.3f} far from nominal 0.80"
