import json

from pmes.config import Config
from pmes.pipeline import run
from pmes.synthetic import generate


def test_pipeline_end_to_end(tmp_path):
    cfg = Config(data_dir=tmp_path / "data", report_dir=tmp_path / "reports", n_bootstrap=200)
    generate(cfg.data_dir, n=6000)
    run(cfg)
    res = json.loads((cfg.report_dir / "results.json").read_text())
    assert res["adjustment"]["post_shock_drift"] > 0
    assert res["strategy"]["delay_0h"]["out_of_sample"]["mean_pnl_pp"] > 0
    for f in ["summary.md", "event_study_prob.png", "lead_lag.png", "trades_oos.png", "events.csv"]:
        assert (cfg.report_dir / f).exists()
