from my_sarima_module import grid_search_sarima, SARIMAGridSearchConfig

config = SARIMAGridSearchConfig(
    p_range=range(0, 3),
    d_range=range(0, 2),
    q_range=range(0, 3),
    P_range=range(0, 2),
    D_range=range(0, 2),
    Q_range=range(0, 2),
    s=12,
)

best_params, results = grid_search_sarima(train_series, test_series, config)