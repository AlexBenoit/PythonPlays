# PythonPlays
Python AI that plays Super Smash Bros. Melee

## Setup

1. Start by cloning the project
2. Create all [virtual environments](#createVirtualEnvs)
3. The project is ready to be executed

## Execution

1. Open Super Smash Bros. Melee
2. Run the project
3. While in the starting countdown of the project, click on the game window to give it focus
4. Let the AI play
5. Press **Q** to stop the program

---

## Project Descriptions

### <a name="CoreComponents"></a>CoreComponents
This project contains the modules common to both the [PythonPlays](#PythonPlays) and 
[PlayRecorder](#PlayRecorder) projects.

### <a name="PythonPlays"></a>PythonPlays
This project is the main project. It contains the code that will play the game. It uses modules
from the [CoreComponents](#CoreComponents) project.

### <a name="PlayRecorder"></a>PlayRecorder
This project is used to recorde a playing session from a user. The data from the recorded session
is used to do an initial supervised training on the [PythonPlays](#PythonPlays) project.

## <a name="createVirtualEnvs"></a>Virtual Environments

1. Make sure you have a default installation of Python installed and its packages are updated
2. Right click **Python Environments** in your project
3. Select **Add Virtual Environment...**
4. Name the environment the same as the project name and add **Env** at the end. (Ex: CoreComponentsEnv)
5. Right click the newly created environment and select **Install from requirements.txt**