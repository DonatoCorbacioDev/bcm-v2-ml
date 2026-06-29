from unittest.mock import MagicMock
from app.services.forecasting import compute_forecast


class FVMock:
    def __init__(self, year, month, amount):
        self.year_value = year
        self.month_value = month
        self.financial_amount = amount


def make_session(rows):
    session = MagicMock()
    session.query.return_value.all.return_value = rows
    return session


def make_org_session(rows):
    session = MagicMock()
    session.query.return_value.filter.return_value.all.return_value = rows
    return session


def test_forecast_empty_db():
    result = compute_forecast(make_session([]), 3)
    assert result == {"historical": [], "forecast": []}


def test_forecast_single_point():
    rows = [FVMock(2024, 1, 10000.0)]
    result = compute_forecast(make_session(rows), 3)
    assert len(result["historical"]) == 1
    assert len(result["forecast"]) == 3
    for f in result["forecast"]:
        assert f["amount"] >= 0.0


def test_forecast_two_points():
    rows = [FVMock(2024, 1, 10000.0), FVMock(2024, 2, 12000.0)]
    result = compute_forecast(make_session(rows), 2)
    assert len(result["historical"]) == 2
    assert len(result["forecast"]) == 2
    for f in result["forecast"]:
        assert f["lower"] <= f["amount"]
        assert f["upper"] >= f["amount"]


def test_forecast_multiple_points_with_stddev():
    rows = [
        FVMock(2024, 1, 10000.0),
        FVMock(2024, 2, 11000.0),
        FVMock(2024, 3, 12000.0),
        FVMock(2024, 4, 13000.0),
    ]
    result = compute_forecast(make_session(rows), 3)
    assert len(result["historical"]) == 4
    assert len(result["forecast"]) == 3
    for item in result["historical"]:
        assert "month" in item
        assert "amount" in item
    for f in result["forecast"]:
        assert f["lower"] <= f["amount"] <= f["upper"]


def test_forecast_aggregates_same_month():
    rows = [FVMock(2024, 1, 5000.0), FVMock(2024, 1, 5000.0)]
    result = compute_forecast(make_session(rows), 1)
    assert len(result["historical"]) == 1
    assert result["historical"][0]["amount"] == 10000.0


def test_forecast_negative_trend_clamps_to_zero():
    rows = [
        FVMock(2024, 1, 50000.0),
        FVMock(2024, 2, 30000.0),
        FVMock(2024, 3, 10000.0),
        FVMock(2024, 4, 1000.0),
    ]
    result = compute_forecast(make_session(rows), 6)
    for f in result["forecast"]:
        assert f["amount"] >= 0.0
        assert f["lower"] >= 0.0


def test_forecast_with_org_id_filters_query():
    rows = [FVMock(2024, 1, 10000.0)]
    session = make_org_session(rows)
    result = compute_forecast(session, 3, org_id=7)
    session.query.return_value.filter.assert_called_once()
    assert len(result["historical"]) == 1


def test_forecast_without_org_id_does_not_filter():
    rows = [FVMock(2024, 1, 10000.0)]
    session = make_session(rows)
    compute_forecast(session, 3, org_id=None)
    session.query.return_value.filter.assert_not_called()


def test_forecast_reliable_flag_false_when_fewer_than_12_months():
    rows = [FVMock(2024, m, 10000.0) for m in range(1, 7)]  # 6 months
    result = compute_forecast(make_session(rows), 3)
    assert result["reliable"] is False


def test_forecast_reliable_flag_true_when_12_or_more_months():
    rows = [FVMock(2023, m, 10000.0) for m in range(1, 13)]  # 12 months
    result = compute_forecast(make_session(rows), 3)
    assert result["reliable"] is True


def test_forecast_single_point_is_not_reliable():
    rows = [FVMock(2024, 1, 10000.0)]
    result = compute_forecast(make_session(rows), 3)
    assert result["reliable"] is False
