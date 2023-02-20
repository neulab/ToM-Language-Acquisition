# ToM-Language-Acquisition
Code and data for the ICLR 2023 paper "Computational Language Acquisition with Theory of Mind".

The script for training new speakers is ```ppo.py```. 
Example usage: ```python ppo.py --total-timesteps 10000000 --supervised-coef 0.01  --game-file-path data/game_file_20.pt --dev-game-file-path data/game_file_dev.pt  --render-html --exp-name HighWeightToMEasy --render-every-N 1000000 --captions-file train_org --track --gamma 1.0 --less-logging --use-tom --beam-size 25 --sigma-decay --tom-weight 1000.0```.

The script for evaluating speakers is ```eval.py```.
Example usage: ```python eval.py --total-timesteps 10000 --supervised-coef 0.01  --game-file-path game_files/test/game_file_test.pt --exp-name RSAEasyCoef3 --captions-file test_org --wandb-project-name RevisionEval --gamma 1.0 --less-logging --use-coco --use-tom --beam-size 25 --sigma 0.0 --seed 517 --tom-weight 1000.0 --pretrained-path [wandb path name] --track```.

```build_new_game_caption.py``` can be used to build new datasets with custom similarity scores, while ```listener_pretraining.py``` can be used to train new listeners.

The conda environment used to run the code can be built from ```environment-spec.txt```.

# Acknowledgement
Code and data were based off of [Hao Zhu](https://github.com/ProKil)'s [InteractGym](https://github.com/ProKil/interactgym) repository.