Held-out sets: test_shard = 100 battles / 6683 plays, test_random = 100 battles / 8090 plays

**Next card, top-1 / top-3 (query right before each play)**

| | test_shard | test_random |
|---|---|---|
| random_legal (deck minus last 4 own plays) | 23.9% / 71.7% | 24.1% / 72.3% |
| most_frequent_legal | 28.6% / 75.1% | 29.9% / 78.9% |
| most_frequent_legal_given_opp_last_card | 30.8% / 76.4% | 32.1% / 80.3% |
| model | 14.5% / 42.2% | 15.5% / 43.7% |
| model + hand mask | 30.8% / 80.5% | 33.1% / 82.6% |

**Tile given the true card: within 1 tile / mean distance (tiles)**

| | test_shard | test_random |
|---|---|---|
| random_tile_where_card_was_ever_placed | 5.9% / 8.69 | 5.7% / 8.60 |
| most_frequent_tile_for_card | 18.8% / 8.24 | 19.4% / 8.08 |
| most_frequent_tile_for_card_and_opp_lane | 20.5% / 7.57 | 20.6% / 7.40 |
| model | 14.4% / 9.22 | 15.0% / 8.99 |
| end-to-end: cond_freq_card + most_frequent_tile | 11.2% / 8.88 | 10.6% / 8.71 |
| end-to-end: model | 9.5% / 10.29 | 9.0% / 9.97 |

**Timing: delay to next own play, MAE (s) / within 1 s; wait-vs-play (play within 1 s) AUC**

| | test_shard | test_random |
|---|---|---|
| constant_median | 3.37 / 25.7% | 3.04 / 27.0% |
| median_by_phase_and_last_actor | 2.95 / 28.3% | 2.71 / 30.7% |
| model | 3.11 / 26.4% | 2.83 / 28.4% |
| wait/play AUC | baseline 0.718, model 0.693 | baseline 0.698, model 0.682 |

**Win probability (all query points): AUC / Brier / ECE**

| | test_shard | test_random |
|---|---|---|
| constant_0.5 | 0.500 / 0.250 / 0.000 | 0.500 / 0.250 / 0.000 |
| logistic_on_elixir_and_play_count_diff | 0.549 / 0.249 / 0.019 | 0.506 / 0.253 / 0.032 |
| model | 0.501 / 0.251 / 0.027 | 0.543 / 0.249 / 0.022 |

**Win-probability AUC by game phase (test_shard)**

| phase | n | model AUC | model ECE | logistic AUC |
|---|---|---|---|---|
| 0-1 min | 2770 | 0.494 | 0.021 | 0.503 |
| 1-2 min | 2618 | 0.498 | 0.030 | 0.536 |
| 2-3 min | 4648 | 0.506 | 0.021 | 0.537 |
| overtime | 4124 | 0.502 | 0.036 | 0.561 |

