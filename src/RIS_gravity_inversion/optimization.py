from __future__ import annotations

import numpy as np
import optuna

import RIS_gravity_inversion.utils as inv_utils


def optimal_buffer(
    top,
    reference,
    spacing,
    interest_region,
    density,
    obs_height,
    target,
    buffer_perc_range=(1, 50),
    n_trials=25,
    density_contrast=False,
    checkerboard=False,
    amplitude=None,
    wavelength=None,
    full_search=False,
):
    """
    Optuna optimization to find best buffer zone width.
    """

    def objective(
        trial,
        top,
        reference,
        spacing,
        interest_region,
        density,
        target,
        obs_height,
        checkerboard,
        density_contrast,
        amplitude,
        wavelength,
    ):
        """
        Find buffer percentage which gives a max decay within the region of interest
        closest
        to target percentage (i.e., target=5 is 5% decay).
        """
        buffer_perc = trial.suggest_int(
            "buffer_perc", buffer_perc_range[0], buffer_perc_range[1]
        )

        max_decay = inv_utils.gravity_decay_buffer(
            buffer_perc=buffer_perc,
            top=top,
            reference=reference,
            spacing=spacing,
            interest_region=interest_region,
            density=density,
            obs_height=obs_height,
            checkerboard=checkerboard,
            density_contrast=density_contrast,
            amplitude=amplitude,
            wavelength=wavelength,
        )[0]

        return np.abs((target / 100) - max_decay)

    # Create a new study
    if full_search is True:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.GridSampler(
                search_space={
                    "buffer_perc": list(
                        range(buffer_perc_range[0], buffer_perc_range[1])
                    )
                }
            ),
        )
    else:
        study = optuna.create_study(
            direction="minimize",
        )

    # Disable logging
    optuna.logging.set_verbosity(optuna.logging.WARN)

    # run optimization
    study.optimize(
        lambda trial: objective(
            trial,
            top,
            reference,
            spacing,
            interest_region,
            density,
            target,
            obs_height,
            checkerboard,
            density_contrast,
            amplitude,
            wavelength,
        ),
        n_trials=n_trials,
    )

    print(study.best_params)

    plot = optuna.visualization.plot_optimization_history(study)
    plot2 = optuna.visualization.plot_slice(study)

    plot.show()
    plot2.show()

    return study, study.trials_dataframe().sort_values(by="value")
