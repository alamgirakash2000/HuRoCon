# HuRoCon: Humanoid Robot Locomotion and Payload Transportation in Construction

## Abstract:
The increasing demand for automation in con-
struction necessitates robotic solutions that can address labor
shortages, safety concerns, and efficiency challenges. Humanoid
robots, with their human-like form factor, are particularly well-
suited for traversing and operating in human-centric construc-
tion environments. We present HuRoCon, an integrated hu-
manoid robotic framework designed for robust locomotion and
payload transportation across diverse construction surfaces.
Our system combines dynamic bipedal walking with effective
payload-carrying mechanisms, enabling seamless transitions
between surfaces such as flat ground, ramps, stairs, narrow
beams, random obstacles, and irregular trenches while carrying
loads of up to 9 lbs. We employ a unified training approach
using reinforcement learning to achieve stable locomotion and
real-time adaptation of walking styles based on surface con-
ditions. Experimental results demonstrate that our framework
enables continuous locomotion at speeds up to 0.42 m/s acrosss
complex construction environments, showcasing the potential
of humanoid robots in construction logistics.


## Code structure:
A rough outline for the repository that might be useful for adding your own robot:
```
LearningHumanoidWalking/
├── envs/                <-- Actions and observation space, PD gains, simulation step, control decimation, init, ...
├── tasks/               <-- Reward function, termination conditions, and more...
├── rl/                  <-- Code for PPO, actor/critic networks, observation normalization process...
├── models/              <-- MuJoCo model files: XMLs/meshes/textures
├── trained/             <-- Contains pretrained model for JVRC
└── scripts/             <-- Utility scripts, etc.
```

## Requirements:
- Python version: 3.7.11  
- [Pytorch](https://pytorch.org/)
- pip install:
  - mujoco==2.2.0
  - [mujoco-python-viewer](https://github.com/rohanpsingh/mujoco-python-viewer)
  - ray==1.9.2
  - transforms3d
  - matplotlib
  - scipy

### To setup the environment
  - `conda create -n HuRoCon python=3.7.11`
  - `conda install protobuf==3.20.3`
  - `conda install pytorch`
  - `pip install ray==1.9.2`
  - `pip install mujoco==2.3.6 mujoco-py==2.1.2.14 mujoco-python-viewer==0.1.4`
  - `pip install dm-control==1.0.13`
  - `pip install transforms3d matplotlib scipy`



## Usage:
Environment names supported:  

| Task Description      | Environment name |
| ----------- | ----------- |
| Construction Walk | 'jvrc_step' |


### **To train:** 
```$ python run_experiment.py train --logdir <path_to_exp_dir> --num_procs <num_of_cpu_procs> --env <name_of_environment>```
  
**Example**: ``` $ python run_experiment.py train --logdir trained/payload --num_procs 8 --env jvrc_step``` 
 



### **To play:** 
```$ PYTHONPATH=.:$PYTHONPATH python scripts/debug_stepper.py --path <path_to_exp_dir>```
**Example**: ``` $ PYTHONPATH=.:$PYTHONPATH python scripts/debug_stepper.py --path ./trained/construction```


#### Some key commands
```To remove the env file - $ rm /tmp/mjcf-export/jvrc_step/jvrc1.xml```
``` The activate the environment - $ conda activate HuRoCon ```

#### **What you should see:**
![Construction Environemnt](demo_images/simulation1.png)

![Humanoid Robot Carrying Payload](demo_images/simulation2.png)

![Locomotion on different Surfaces](demo_images/all_surface.png)

![Challenging Surfaces](demo_images/fig3.png)


#### **Inspired from:**
[LearningHumanoid](https://github.com/rohanpsingh/LearningHumanoidWalking)
