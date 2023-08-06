# Piano transcription WebUI

## Installation

1. Download `piano_transcription_webui-0.0.1-py3-none-any.whl`
1. Command `pip install piano_transcription_webui-0.0.1-py3-none-any.whl`
1. If you are using conda environment, enter `conda install ffmpeg` to install FFmpeg.
   If not, download `ffmpeg.exe` from [https://ffmpeg.org/] and add its path to
   system environment variable.

## How to use

1. Download model `note_F1=0.9677_pedal_F1=0.9186.pth`
1. Command `transui`
1. Access `https://127.0.0.1:7860` in the browser(The URL depends on the actual situation)
1. Fill in information and press SUBMIT button
1. Wait
1. Download the .mid file