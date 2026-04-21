from derma_jepa.metrics import binary_metric_summary


def test_binary_metric_summary_reports_perfect_separation() -> None:
    summary = binary_metric_summary(
        [0, 0, 1, 1],
        [0.05, 0.10, 0.90, 0.95],
        bootstrap_samples=50,
        ci_level=0.95,
        fixed_tpr=0.80,
        seed=123,
    )

    assert summary.auroc == 1.0
    assert summary.auprc == 1.0
    assert summary.auroc_ci_low == 1.0
    assert summary.auroc_ci_high == 1.0
    assert summary.fpr_at_fixed_tpr == 0.0
