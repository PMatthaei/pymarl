```diff
- This is a fork of https://github.com/oxwhirl/pymarl !
```

# Python MARL framework

PyMARL is [WhiRL](http://whirl.cs.ox.ac.uk)'s framework for deep multi-agent reinforcement learning and includes implementations of the following algorithms:
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)
- [**QTRAN**: QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)

PyMARL is written in PyTorch and uses [SMAC](https://github.com/oxwhirl/smac) as its environment.

## Installation instructions

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 

Before running experiments read every chapter if you plan to run experiments with saved models and/or replay output carefully.

### Configuration
Configuration files act as defaults for an algorithm or environment. 

Configuration files are all located in `src/config`.
- `--config` refers to the [algorithm config files](https://github.com/PMatthaei/pymarl/tree/master/src/config/algs) in `src/config/algs`
- `--env-config` refers to the [environment config files](https://github.com/PMatthaei/pymarl/tree/master/src/config/envs) in `src/config/envs`

The previous config files used for the SMAC Beta have the suffix `_beta`.

### Run via Commandline
Make sure you are in directory ``pymarl`` and have a configured ``python3`` interpreter.
```shell
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z
```

### Run via Docker
To run experiments using the Docker container:
```shell
bash run.sh python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z
```
After execution you will be promted to choose if you would like to run the experiment on all GPUs or on the CPU.
Ensure you have drivers and CUDA installed before choosing GPU experiment runs.

#### Docker Version >= 19.03
Since docker 19.03 there is built-in support for nvidia GPUs. Thus no additional installation of nvidia-docker is required.
For more information see: https://github.com/NVIDIA/nvidia-docker.

Instead you can use `docker run --gpus all ...` to run your image on _every_ graphic card installed and supported.
Thus no GPU IDs have to be passed as argument if you want to train on all connected graphic cards. 
If necessary GPU IDs can be retrieved via `nvidi-smi` on linux.

##### Examples (Docker >=19.03)**

###### Task
Run COMA in 15m environment on all GPUs (see argument behind run.sh) and save model (at default timestep rate and default directory).

**Ubuntu 18.04:** 

```shell
bash run.sh all python3 src/main.py --config=coma --env-config=sc2 with env_args.map_name=15m save_model=True
```
Saving a replay file:

```shell
bash run.sh all python3 src/main.py --config=coma --env-config=sc2 with env_args.map_name=15m save_replay=True runner=episode
```
 
**Windows 10** (not recommended/supported):

**Note:** Since the current run.sh is configured for ubuntu the following line should achieve the same as the above.
```
docker run --gpus all -v "$(pwd):/pymarl" -t pymarl:1.0 python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=15m save_model=True save_replay=True
```

### Run via PyCharm/Virtual environment

1. Add `SC2PATH` to your environment variables under `Edit Configurations` where the value is pointing to the directory you have installed SC2 binaries.
2. Create virtual environment (conda etc)
3. Add dependencies
4. Run main.py with example parameters: `--config=qmix --env-config=sc2 with env_args.map_name=2s3z`
### Results
All results will be stored in the `Results` folder under ``results/sacred``.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model=True`, which is set to `False` by default. 
The frequency of saving models can be adjusted using `save_model_interval` configuration. 
Models will be saved in the result directory, under the folder called *models*. 
The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

**Example for COMA (Docker >=19.03):**

```shell
bash run.sh ... checkpoint_path=results/models/<your model folder> save_model=True
```

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

### Model collections

The following repository is holding a lot of models already obtained: https://github.com/PMatthaei/pymarl-results

In order to use these results, clone the repo and copy the master folder into the results folder before mounting it into a docker container.

```shell
bash run.sh python3 src/main.py --config=coma --env-config=sc2 with env_args.map_name=3m checkpoint_path=/home/gaming-ubuntu/Projects/pymarlresults/3m_coma/model save_replay=True runner=episode batch_size=1 batch_size_run=1
```

### Generating a replay file
`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. 
The checkpoint argument is mandatory to tell StarCraft which model to use. 
Reference a model from the folder *models* (see above). 
Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. 
Please make sure to **use the episode runner if you wish to save a replay**, i.e., `runner=episode`. 
The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

### Watching a replay

Follow the setup guide for PySC2: https://github.com/deepmind/pysc2
1. Download SC2 binaries
2. Unpack to a directory of your choice
3. Set SC2PATH environment variable to that directory
4. Download SMAC Maps
5. Unpack and copy content to /Maps

**Note:** Replays within the Starcraft Engine cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client. PySC2 is only serving a alternate Interface which does not render Starcraft Assets.

See: https://github.com/oxwhirl/smac/issues/22

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay <NAME>.SC2Replay
```

Run in docker (>=19.03): 

**Note:** This will show nothing. Removing `--norender` results in an error
```shell
bash run.sh python3 -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay <NAME>.SC2Replay
```

## Documentation/Support

This is an adapted documentation for this repos specific needs. For the original documentation please visit this forks base repository.

## Citing PyMARL 

If you use PyMARL in your research, please cite the [SMAC paper](https://arxiv.org/abs/1902.04043).

*M. Samvelyan, T. Rashid, C. Schroeder de Witt, G. Farquhar, N. Nardelli, T.G.J. Rudner, C.-M. Hung, P.H.S. Torr, J. Foerster, S. Whiteson. The StarCraft Multi-Agent Challenge, CoRR abs/1902.04043, 2019.*

In BibTeX format:

```tex
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```

## License

Code licensed under the Apache License v2.0
