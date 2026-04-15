from fit_3d import get_estimator_class, get_rule_class, load_data, main, run_experiment

Rule = get_rule_class({'model': {'type': 'curve'}})


if __name__ == "__main__":
    main()
