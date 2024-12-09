# Dysarthria

# Hybrid ASR-TTS
## Setup
We recommend installing NeMo in a fresh Conda environment.
`conda create --name nemo python==3.8`
`conda activate nemo`

# Build NeMo Container
Run the following command to use NVIDIA PyTorch container version 22.11-py3.
`docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo --shm-size32g \
-p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:22.11-py3`

# Install Dependencies
Run the following commands _inside the docker container_ to ensure you have the required dependencies in the container environment.
`pip install nemo_toolkit['asr']`
`pip install nemo_text_processing`
`pip install nemo_toolkit['tts']`
`pip install wandb`

# Training the ASR Model
Run `python asrtts_zh.py` for training.
During training, it fine-tunes the existing end-to-end ASR model and saves the checkpoints of both the trained hybrid end-to-end ASR model and the trained end-to-end ASR model. Each training is set to 100 epochs. 

# ASR Inference
`asrtts_inference.py`
During inference, the predicted text from the end-to-end ASR model is saved as a string variable and parsed to the end-to-end TTS model to generate the synthetic speech.

# Audio2Face Avatar Animation Stream
This document details the steps for setting up Avatar Animation Stream pipeline with the following microservices to be run as separate Docker containers:
- Audio2Face-3D microservice
- Animation Graph microservice
- Omniverse Renderer microservice

You will also need to run the Gstreamer client for visualization and audio purposes.

### Setup
The setup for this follows the documentation [here](https://docs.nvidia.com/ace/latest/workflows/animation_pipeline/docker_with_omniverse_renderer.html).
The following details the steps followed according to the documentation and any deviation from the documentation is included. Screenshots are included.

### Prerequisites
This application uses the microservices and clients as mentioned above, hence the softwares in the Prerequisites section must be installed.

### Minimum GPU Requirements
Follow the [Hardware Requirements section](https://docs.nvidia.com/ace/latest/workflows/animation_pipeline/docker_with_omniverse_renderer.html#hardware-requirements) of the documentation
to ensure you meet the hardware requirements to run all the microservices. It may be advisable to allow additional hardware usage if you are running all microservices on the same device.

### Download Avatar Scene from NGC
Download the Avatar Scene (2 options available) and make sure to store it in your main work directory as the directory name mentioned `default-avatar-scene_v1.0.0`. The microservices will need to refer to this folder later.

### Run Animation Graph Microservice
![image](https://github.com/user-attachments/assets/13a48390-522f-4b41-b75a-1cf98fc15015)
Run the command as per the documentation and the docker container should have the final log line `app ready`.

### Run Omniverse Renderer Microservice
In a separate terminal in your main working directory, run the docker run command to start the Omniverse Renderer Microservice, making sAudio2Face Avatar Animation Streamure to replace the `<path-to-avatar-scene-folder>` appropriately.
![image](https://github.com/user-attachments/assets/cafcf21b-40a9-41da-8342-c2ef60a64ada)
As the document stated, the service is ready when you see the line `[SceneLoader] Assets loaded`

### Run Audio2Face-3D Microservice
Open a separate terminal in your main working directory. Do not follow the documentation for [this section](https://docs.nvidia.com/ace/latest/workflows/animation_pipeline/docker_with_omniverse_renderer.html#run-audio2face-3d-microservice).
Instead, clone the [NVIDIA/ACE repository](https://github.com/NVIDIA/ACE) and follow the setup instructions [here](https://github.com/NVIDIA/ACE/tree/main/microservices/audio_2_face_microservice/quick-start).
This will run the Audio2Face-3D Microservice with the default configurations.

![image](https://github.com/user-attachments/assets/b73ee1a7-e8be-4b6e-9afe-9701c3210bf3)
![image](https://github.com/user-attachments/assets/910398bd-ac4d-4867-8d4a-bacb96c443ac)
The docker compose file will create and run 3 services. The Audio2Face-3D Microservice is running successfully when you see the `[  global  ] [info] Running...` log line.

You may need to change the image paths for each docker service in the `docker-compose.yml` file as mentioned on [this page](https://docs.nvidia.com/ace/latest/modules/a2f-docs/text/getting_started/resource_deprecation_warning.html).

The docker compose in quickstart as above has been configured to do the same as per the [original documentation](https://docs.nvidia.com/ace/latest/workflows/animation_pipeline/docker_with_omniverse_renderer.html#run-audio2face-3d-microservice).

### Setup & Start Gstreamer
In two separate terminals, follow [the documentation](https://docs.nvidia.com/ace/latest/workflows/animation_pipeline/docker_with_omniverse_renderer.html#setup-start-gstreamer) to run the Gstreamer video receiver and audio receiver separately.

![image](https://github.com/user-attachments/assets/17c00e60-ac43-4e9d-bf49-a0c7a5916be0)
_Logs after running command for video receiver_

![image](https://github.com/user-attachments/assets/acfef4ec-639b-479e-a97e-e31a096d57be)
_Logs after running command for audio receiver_

Running the video receiver and audio receiver commands will only show some logs until the next step is completed.

### Connect Microservices through Common Stream ID
Follow [the documentation](https://docs.nvidia.com/ace/latest/workflows/animation_pipeline/docker_with_omniverse_renderer.html#connect-microservices-through-common-stream-id) to link all the services up.
![image](https://github.com/user-attachments/assets/8359dd2d-313b-46f2-8077-c1deb1e453ff)
A new window should appear with an avatar moving slightly.

### Checkpoint
At this point, you should have 5 terminals running the following separately:
- Audio2Face-3D microservice
- Animation Graph microservice
- Omniverse Renderer microservice
- Gstreamer video receiver
- Gstreamer audio receiver

### Test Animation Graph Interface
In a new terminal in your main working directory, follow [this section](https://docs.nvidia.com/ace/latest/workflows/animation_pipeline/docker_with_omniverse_renderer.html#test-animation-graph-interface) to view changes to the avatar on screen. Documentation to the available posture / position / gesture / facial gesture are also included in the documentation.

https://github.com/user-attachments/assets/d02ec781-1677-4673-b764-f453d58b2bbf

_Example: Run the command `curl -X PUT -s http://localhost:8020/streams/$stream_id/animation_graphs/avatar/variables/gesture_state/The_Robot` in main terminal to make the avatar do a short robot dance_

### Test Audio2Face-3D
Follow [this section](https://docs.nvidia.com/ace/latest/workflows/animation_pipeline/docker_with_omniverse_renderer.html#test-audio2face-3d) to send your own audio file to the Avatar Animation Stream. You need to download the ACE repository and setup the python script.

https://github.com/user-attachments/assets/60483c87-5374-4e38-9beb-13dc45276459

_Running the command as given in the documentation results in the avatar speaking as per the input audio file_

#### Mandarin Audio Input
https://github.com/user-attachments/assets/bf670341-a33f-42e6-b54b-db425f3ba3a4

_Running the command with a sample mandarin input audio file_

Change the command accordingly by replacing `<filepath_to_custom_audio_file>` with custom input audio files for the avatar to speak with: `python3 validate.py -u 127.0.0.1:50000 -i $stream_id <filepath_to_custom_audio_file>`.

## Customisations
To customise your own avatar and scene, you need to use the Avatar Configurator and follow the instructions [here](https://docs.nvidia.com/ace/latest/modules/avatar_customization/Avatar_Configurator.html#avatar-configurator) for setup instructions.

![image](https://github.com/user-attachments/assets/2ea52c50-55a8-4e77-a825-72893bb1796e)
_The Avatar Configurator will take some time to load_

![image](https://github.com/user-attachments/assets/9b23a968-5f68-4eca-889a-a73e2be4ef2b)
_A default avatar will appear, indicating the Avatar Configurator is ready for use_

![image](https://github.com/user-attachments/assets/01de3e88-72d8-4bf4-8c75-98d2e69c69aa)
_Make changes to the avatar and scene in the Properties panel on the right side of the window. Click on Save Scene at the bottom right when satisfied._

![image](https://github.com/user-attachments/assets/35ffe128-0e08-4396-a246-c3c8b4650a69)
_Save scene will save all necessary files to the /exported folder. You will use this folder in your Audio2Face Avatar Animation Stream in the next step._

### Using your customised Avatar and Scene
Replace the path to your newly created and saved avatar and scene folder accordingly.

1. Run Animation Graph Microservice by changing the command to `docker run -it --rm --gpus all --network=host --name anim-graph-ms -v <path_to_your_new_avatar_and_scene_folder>:/home/ace/asset nvcr.io/nvidia/ace/ia-animation-graph-microservice:1.0.2`.
2. Run Omniverse Renderer Microservice by running the command to `docker run --env IAORMS_RTP_NEGOTIATION_HOST_MOCKING_ENABLED=true --rm --gpus all --network=host --name renderer-ms -v <path_to_your_new_avatar_and_scene_folder>:/home/ace/asset nvcr.io/nvidia/ace/ia-omniverse-renderer-microservice:1.0.5`
3. You may run the remaining microservice and the Gstreamer audio and video listeners same as the above. When you have connected the microservices with the Common Stream ID, the application will be running your newly created avatar and scene.
4. You may now invoke the same python command to invoke the validate.py script as mentioned in [this section of the documentation] (https://docs.nvidia.com/ace/latest/workflows/animation_pipeline/docker_with_omniverse_renderer.html#test-audio2face-3d) to make your avatar speak with your audio files.

![image](https://github.com/user-attachments/assets/61ed14b7-c89f-46be-b3bb-2524ec039ae2)
_The application now starts with your newly created avatar and scene_

https://github.com/user-attachments/assets/17939309-88a7-48f4-a152-3ea716412a16

https://github.com/user-attachments/assets/d2a5ba2b-306c-4a4b-9c9f-bb4b0a3862ec

_Your newly created avatar and scene are similarly controlled as before_

