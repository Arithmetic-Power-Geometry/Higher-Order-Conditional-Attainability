# Stopping-score clarification

The controlled synthetic benchmark permits at most 10 intervention steps and uses decision-sufficiency threshold epsilon = 0.30.

For a run that reaches the threshold within the budget, the observed stopping time is

`tau_epsilon = min{t : Omega_T(C_t) <= epsilon}`.

For summaries that include both successful and unsuccessful runs, the benchmark endpoint field historically named `tau_eps` uses the capped/censored score

`tau_tilde_epsilon = tau_epsilon` when the threshold is reached within steps 0--10, and `tau_tilde_epsilon = 11` otherwise.

Thus a value of 11 is a censoring code for failure to reach decision sufficiency within the 10-intervention budget; it is not an observed eleventh intervention. Success rate is reported separately and must be read alongside the capped score.

For manuscript reporting, the endpoint means 9.1227 (OICA), 10.0532 (Graph), 10.5486 (ClassSize), and 9.6782 (Random) are therefore mean capped stopping scores, not uncensored mean stopping times.

The cumulative-cost means from the same workflow output are 12.0319 (OICA), 11.8392 (Graph), 12.0772 (ClassSize), and 11.9871 (Random).

This clarification changes terminology only. It does not alter generated trajectories, endpoint values, paired bootstrap results, figures, or the benchmark's scope boundary. CSuite and Sachs were not executed in this benchmark.
