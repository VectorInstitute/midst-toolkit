The data processing pipeline is right now specific to the MIDST competition provided
resource folders.

Step 1: Collect all the train data from all the attack types (every train folder provided by the challenge). "population_all.csv" includes 867494 data points.

Step 2: Collect all the challenge data points from train, dev and final folders of tabddpm_black_box.  `challenge_points_all.csv` includes 13896 data points.

Step 3: Save population data without and with challenge points. `population_all_no_challenge.csv` includes 855644 data points, and `population_all_with_challenge.csv` includes 869540 data points.

`population_all_with_challenge.csv` is used to create real train and test data (referred to as `Population/Subset (Real Data)` in the Figure). Note that a random subset of 40k data points are sampled from `population_all_with_challenge.csv` and used as population (or real data).

To run the whole data processing pipeline run `process_split_data.py`. It reads data from `ensemble_mia/data/midst_data_all_attacks`, and populates `ensemble_mia/data/population_data` and `ensemble_mia/data/attack_data` folders.
