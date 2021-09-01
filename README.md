# Pajlada TTS

Stripped down fork of ForwardTacotron (https://github.com/as-ideas/ForwardTacotron) with pretrained weights for Pajlada's (https://github.com/pajlada) voice.

## ⚙️ Installation

Make sure you have:

* Python >= 3.7

Install espeak as phonemizer backend, (for macOS use brew):
```
sudo apt-get install espeak
```
For windows you can use the installer (https://github.com/espeak-ng/espeak-ng/releases), then make sure that in your path PHONEMIZER_ESPEAK_PATH points to espeak-ng.exe

Then install the rest with pip:
```
pip install -r requirements.txt
```
If you aren't going to use CUDA, you can install the smaller CPU only torch version from https://pytorch.org/get-started/locally/

Get the pretrained weights and extract the checkpoints folder to the root where gen_tacotron.py is:
```
https://drive.google.com/file/d/13I_x2bU6rXTqqIe9Lj8lZyS5OaW5TId-/view
```

## ❓ Examples 

Generate all sentences from sentences.txt:
```
python gen_tacotron.py --config config.yaml wavernn
```

Generate given sentence with lower quality Griffin-Lim vocoder, forcing CPU use:
```
python gen_tacotron.py --config config.yaml --input_text "Freedom means not having a master." --cpu griffinlim 
```

Autoregressive models can get stuck in loops if you try to generate too tricky or large sentences. Ending all sentences with a full stop helps.

## Acknowlegements

* [https://github.com/pajlada](https://github.com/pajlada)
* [https://github.com/as-ideas/ForwardTacotron(https://github.com/as-ideas/ForwardTacotron)]
* [https://github.com/keithito/tacotron](https://github.com/keithito/tacotron)
* [https://github.com/fatchord/WaveRNN](https://github.com/fatchord/WaveRNN)

## Copyright

See [LICENSE](LICENSE) for details.
