# Dysarthria

# Hybrid ASR-TTS Setup


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
In a separate terminal in your main working directory, run the docker run command to start the Omniverse Renderer Microservice, making sure to replace the `<path-to-avatar-scene-folder>` appropriately.
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
To customise your own avatar and scene, you need to use the Avatar Configurator and follow the instructions [here](https://docs.nvidia.com/ace/latest/modules/avatar_customization/Avatar_Configurator.html#avatar-configurator).
