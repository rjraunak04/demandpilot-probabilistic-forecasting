import numpy as np

from demandpilot.evaluation.metrics import evaluate_forecast


def main() -> None:
    actual = np.array([0.0, 2.0, 5.0, 10.0, 0.0, 4.0])
    predicted = np.array([0.0, 3.0, 4.0, 8.0, 1.0, 5.0])

    metrics = evaluate_forecast(actual, predicted)

    assert metrics.wape >= 0.0
    assert metrics.mae >= 0.0
    assert metrics.rmse >= metrics.mae
    assert 0.0 <= metrics.smape <= 2.0

    print("DemandPilot Day-03 CP-02 forecast metrics PASSED")
    print(f"WAPE:  {metrics.wape:.6f}")
    print(f"MAE:   {metrics.mae:.6f}")
    print(f"RMSE:  {metrics.rmse:.6f}")
    print(f"sMAPE: {metrics.smape:.6f}")


if __name__ == "__main__":
    main()