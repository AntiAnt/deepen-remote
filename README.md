# deepen-remote
Remotely processes media using AWS infra

## Setup
Env setup using mamba
```
mamba env create -f environment.yml -y
```
activate env
```
mamba activate deepen-remote
```
pip install using uv
```
uv pip install -r requirements.txt
```

## Run
Add a run config **do not commit config** as secrets maybe be included.


run audio transcription and summary
```
python run deepen_remote.py --relic-name <relic name> --relic-type <relic type> --storage-name <relic config storage name>
```