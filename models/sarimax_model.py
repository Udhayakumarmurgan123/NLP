params = SARIMAGridSearchParams(
    p_range=range(0, 3),
    d_range=range(0, 2),
    q_range=range(0, 3),
    P_range=range(0, 2),
    D_range=range(0, 2),
    Q_range=range(0, 2),
    s=12,
)

best, all_results = grid_search_sarima(train_series, test_series, params)