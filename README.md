# Gesture-Based Collaborative Whiteboard

This project demonstrates a collaborative whiteboard system combining gesture-based interaction, FPGA acceleration, and cloud communication. The system enables real-time multi-user interaction while maintaining low latency through local processing of gesture data.

[![Watch the video](https://img.youtube.com/vi/ka_kZNjIJMg/0.jpg)](https://youtu.be/ka_kZNjIJMg)


## Requirements

Before running our project, here are a short list of packages that you need to install in order to run the code:

- numpy
- opencv-python
- mediapipe
- flask
- python-socketio
- aiohttp
- boto3
- botocore
- torch
- scikit-learn
- matplotlib
- pynq (for the jupyter notebook)

These can be installed using pip install.

## PYNQ Setup

You also need to need to copy the folder `Info-Proc-CW/mlp` to the PYNQ-Z1 board and run the Jupyter Notebook called `Updated.ipynb`.

## Server Setup

Also in order to run our server and serve the html files you need to be able to ssh into our EC2 instance (To Sarim and Aaron, We have emailed the PEM key to you). The PEM key must be placed in `Info-Proc-CW/apps/whiteboard_server`.

## Scripts

We have created bash scripts to make the process easier once the PEM key is in the repository, simply run from the root directory:

- `./scripts/host.sh` (to host the server and serve the html files)
- `./scripts/kill.sh` (corresponding script to kill processes on the EC2 instance)
- `./scripts.run_clients.sh` (captures webcam footage from your laptop, scripts that send and receive data to the PYNQ PS, forwards shapes to the server)