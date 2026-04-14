# SARIMA stasjonaritet

| label | d | D | n_obs | adf_stat | p_value | stationary | critical_5pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ingen differensiering | 0 | 0 | 45 | -4.0529 | 0.0012 | True | -2.9299 |
| Første differense | 1 | 0 | 44 | -9.4707 | 0.0000 | True | -2.9315 |
| Sesongdifferense (12) | 0 | 1 | 33 | -2.3264 | 0.1636 | False | -2.9605 |
| Første + sesongdifferense | 1 | 1 | 32 | -0.5335 | 0.8853 | False | -3.0131 |
