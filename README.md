# Reinforcement Learning of Peg Insertion Robot Arm Agent with Multimodal Sensor Fusion 

A prelimilary version of the python implementation. The code is not well organized currently. 

We will release a nicer version later. ( \_(:з」∠)\_ painful final exams...)

The idea of this project is inspired by the papers written by Michelle Lee, Yuke Zhu and etc.:

Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks: https://arxiv.org/abs/1810.10191

Making Sense of Vision and Touch: Learning Multimodal Representations for Contact-Rich Tasks: https://arxiv.org/abs/1907.13098

Some of the code are taken from their implementation: https://github.com/stanford-iprl-lab/multimodal_representation

The PPO trainer deployed is borrowed from the Assignment 5 of IERG5350 - Reinforcement Learning: https://github.com/cuhkrlcourse/ierg5350-assignment

The borrowed code has been modified to fit in this application.

The simulation environment is constructed using pybullet. Basicly, it contains a kuka robot arm, a cover box and a button inside the box. There is a hole at the upper side of the box. The kuka's end-effector(the peg) can only press the button by inserting the peg into the hole. The agent will gain 10 reward with touching the cover box and 50 reward with pressing the button. A detailed explainantion will be released later (maybe not).

## requirements
`pip install -r requirements.txt`

## train the agent
`python train_peg_insertion.py`


## collect the multimodal dataset for encoder pre-train
`python environments/kuka_peg_env.py`
\[Note\] You will be able to get more data by changing the random seed.

## pre-train the fusion encoder
`python multimodal/train_my_fusion_model.py`
\[Note\] Specify the path to the root directory of multimodal dataset



