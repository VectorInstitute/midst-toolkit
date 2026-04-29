Dataset:

https://drive.google.com/file/d/1HTgfgeL5GXc8uAGfeQirJrUynK7vFeyb/view?usp=sharing


python -m examples.tabsyn.train --config-path=./ensemble_attack

python -m examples.tabsyn.synthesize --config-path=./ensemble_attack

python -m examples.tabsyn.ensemble_attack.make_challenge_dataset

python -m examples.tabsyn.ensemble_attack.train_attack_model

python -m examples.tabsyn.ensemble_attack.test_attack_model
