from unittest.mock import MagicMock, patch
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


def test_forecast_with_manager_id_joins_and_filters_by_manager():
    rows = [FVMock(2024, 1, 10000.0)]
    session = MagicMock()
    session.query.return_value.join.return_value.filter.return_value.all.return_value = rows

    result = compute_forecast(session, 3, org_id=7, manager_id=42)

    session.query.return_value.join.return_value.filter.assert_called_once()
    assert len(result["historical"]) == 1


def test_forecast_manager_id_takes_precedence_over_org_id():
    # Regression test: when both are given (a MANAGER caller), manager_id must
    # be the one applied — org_id alone would leak the whole org's data.
    rows = [FVMock(2024, 1, 10000.0)]
    session = MagicMock()
    session.query.return_value.join.return_value.filter.return_value.all.return_value = rows

    compute_forecast(session, 3, org_id=7, manager_id=42)

    session.query.return_value.filter.assert_not_called()


def months_of_history(n):
    return [FVMock(2022 + i // 12, i % 12 + 1, 10000.0) for i in range(n)]


def test_forecast_reliable_flag_false_when_fewer_than_24_months():
    result = compute_forecast(make_session(months_of_history(6)), 3)
    assert result["reliable"] is False


def test_forecast_reliable_flag_false_just_below_threshold():
    # Regression test: 12-23 months used to report reliable=True while
    # Prophet was still fit without yearly seasonality (threshold 24).
    result = compute_forecast(make_session(months_of_history(23)), 3)
    assert result["reliable"] is False


def test_forecast_reliable_flag_true_when_24_or_more_months():
    result = compute_forecast(make_session(months_of_history(24)), 3)
    assert result["reliable"] is True


@patch("app.services.forecasting.Prophet")
def test_seasonality_and_reliable_share_one_threshold(prophet_cls):
    # The two decisions must flip at the same history length, never apart.
    from app.services.forecasting import MIN_MONTHS_FOR_SEASONALITY
    prophet_cls.return_value.predict.side_effect = RuntimeError("skip fit")
    for n in (MIN_MONTHS_FOR_SEASONALITY - 1, MIN_MONTHS_FOR_SEASONALITY):
        result = compute_forecast(make_session(months_of_history(n)), 3)
        yearly = prophet_cls.call_args.kwargs["yearly_seasonality"]
        assert yearly is result["reliable"]


def test_forecast_single_point_is_not_reliable():
    rows = [FVMock(2024, 1, 10000.0)]
    result = compute_forecast(make_session(rows), 3)
    assert result["reliable"] is False


def test_forecast_prophet_exception_falls_back_to_flat():
    rows = [FVMock(2024, m, 10000.0) for m in range(1, 4)]
    with patch("app.services.forecasting._prophet_forecast", side_effect=RuntimeError("Stan error")):
        result = compute_forecast(make_session(rows), 2)
    assert len(result["forecast"]) == 2
    for f in result["forecast"]:
        assert f["amount"] >= 0.0
        assert "lower" in f
        assert "upper" in f
