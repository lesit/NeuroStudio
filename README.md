# Neuro Studio
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Neuro Studio is GUI based Deep-learning Framework. It has been developed by Jae Wook Kim since 2015.

Check out the [project site](http://www.ainhuman.com) for all the details like
- Brief description

Some screenshots are below
- Create model
![](docs/img/NeuroStudio-create_model.png)

- Setup displays for simulation
![](docs/img/NeuroStudio-setup_sim.png)

- Visualization Tool for simulation
![](docs/img/NeuroStudio-simulating.png)

[Sample Video](https://www.youtube.com/watch?v=WvbD-ejz2NI)

## Future plans
- Fix some bugs including gui related bugs
- Test with LSTM
- Add some data preprocessor
- Add an environment for Reinforcement Learning, including the latest algorithms such as A3C as well as DQN
- Add an environment for GAN, including the latest algorithms such as Progressive growing of GANs
- Add more layer and loss, optimizer algorithms
- Develope a Virtual Machine to run a model by created Neuro Studio. It will run any computer machine.

## Environments
This source must be compiled whth CUDA support.
So, you need to install CUDA 9.0, cudnn v7.x libraries.

All code except file handling and gui parts of Neuro Studio is developed in Standard C++.
It currently works in Windows 7 and above, but if you modify only a few parts, such as gui, you will be able to work on linux as well.

## License
Neuro Studio is BSD-style licensed, as found in the LICENSE file.
