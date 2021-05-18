import multiprocessing as mp
import numpy as np
from RollingWindow import RollingWindow


def single_experiment_hyperopt(par):
    return single_experiment(**par)


def call_function_with_args_and_kwargs(f, args, kwargs):
    return f(*args, **kwargs)


def single_experiment(
    dataframe,
    frequency,
    train_end,
    forecast_column,
    forceast_horizon,
    forecast_metric,
    forecast_function,
    **kwargs
):
    rw = RollingWindow(
        dataframe,
        frequency,
        train_end,
        forceast_horizon,
    )

    with mp.Pool(mp.cpu_count() - 2) as p:
        results = p.starmap(
            call_function_with_args_and_kwargs,
            (
                (forecast_function, (train, test, forecast_column), kwargs)
                for train, test in rw
            ),
        )

    output = np.array([result[0][forecast_metric] for result in results])
    output = output.mean()
    return output
