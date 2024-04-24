import itertools
import solve_recaptcha

# Define the range of values for each parameter
#uncomment for parameter tuning
"""
threshold_values = [0.2]
use_top_n_strategy_values = [False]
n_values = [2, 3, 4]
natural_mouse_movement = [True, False]

# Generate all combinations of these parameters
parameter_combinations = []

for use_top_n_strategy in use_top_n_strategy_values:
    if use_top_n_strategy:
        for n in n_values:
            for threshold in threshold_values:
                for natural_mouse in natural_mouse_movement:
                    parameter_combinations.append((threshold, use_top_n_strategy, n, natural_mouse))
    else:
        for threshold in threshold_values:
            for natural_mouse in natural_mouse_movement:
                parameter_combinations.append((threshold, use_top_n_strategy, None, natural_mouse))

# Iterate over all combinations of parameters
for parameters in parameter_combinations:
    threshold, use_top_n_strategy, n, natural_mouse = parameters

    # Set the variables in solve_recaptcha
    variables = {
        'THRESHOLD': threshold,
        'USE_TOP_N_STRATEGY': use_top_n_strategy,
        'N': n,
        'NATURAL_MOUSE_MOVEMENT': natural_mouse,
    }
    solve_recaptcha.set_variables(variables)

    # Run solve_recaptcha and handle any exceptions
    try:
        solve_recaptcha.run()
        solve_recaptcha.reset_globals()
    except Exception as e:
        print(f"An error occurred with parameters {parameters}: {e}")
"""

for i in range(1, 50):
    solve_recaptcha.run()
    solve_recaptcha.reset_globals()