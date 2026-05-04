Dataset:

https://drive.google.com/file/d/16XCa63eD2dZ1bddhgRbGFuzAuuMlto9P/view?usp=sharing


python -m examples.tabsyn.train --config-path=./ensemble_attack

python -m examples.tabsyn.synthesize --config-path=./ensemble_attack

python -m examples.tabsyn.evaluate --config-path=./ensemble_attack

python -m examples.tabsyn.ensemble_attack.make_challenge_dataset

python -m examples.tabsyn.ensemble_attack.train_attack_model

python -m examples.tabsyn.ensemble_attack.test_attack_model

python -m examples.gan.ensemble_attack.compute_attack_success

python -m midst_toolkit.evaluation.quality.scripts.midst_alpha_precision_eval \
  --synthetic_data_path examples/tabsyn/ensemble_attack/results/trans/synthetic_data/trans_synthetic.csv \
  --real_data examples/tabsyn/ensemble_attack/results/trans_sampled.csv \
  --meta_info_path examples/tabsyn/ensemble_attack/data/meta_info.json \
  --save_directory examples/tabsyn/ensemble_attack/results/
