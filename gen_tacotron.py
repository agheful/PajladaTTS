import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import numpy as np

from models.fatchord_version import WaveRNN
from models.tacotron import Tacotron
from utils.display import simple_table
from utils.dsp import DSP
from utils.files import read_config
from utils.paths import Paths
from utils.text.cleaners import Cleaner
from utils.text.tokenizer import Tokenizer


def load_taco(checkpoint_path: str) -> Tuple[Tacotron, Dict[str, Any]]:
    print(f'Loading tts checkpoint {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    config = checkpoint['config']
    tts_model = Tacotron.from_config(config)
    tts_model.load_state_dict(checkpoint['model'])
    print(f'Loaded taco with step {tts_model.get_step()}')
    return tts_model, config


def load_wavernn(checkpoint_path: str) -> Tuple[WaveRNN, Dict[str, Any]]:
    print(f'Loading voc checkpoint {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    config = checkpoint['config']
    voc_model = WaveRNN.from_config(config)
    voc_model.load_state_dict(checkpoint['model'])
    print(f'Loaded wavernn with step {voc_model.get_step()}')
    return voc_model, config


if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--input_text', '-i', default=None, type=str, help='[string] Type in something here and TTS will generate it!')
    parser.add_argument('--checkpoint', type=str, default=None, help='[string/path] path to .pt model file.')
    parser.add_argument('--config', metavar='FILE', default='config.yaml', help='The config containing all hyperparams. Only'
                                                                                'used if no checkpoint is set.')
    parser.add_argument('--steps', type=int, default=1000, help='Max number of steps.')
    parser.add_argument('--cpu', action='store_true', help='Force CPU use.')
    parser.add_argument('--voc_checkpoint', type=str, default=None, help='[string/path] Load in different WaveRNN weights')


    # name of subcommand goes to args.vocoder
    subparsers = parser.add_subparsers(dest='vocoder')
    wr_parser = subparsers.add_parser('wavernn')
    wr_parser.add_argument('--overlap', '-o', default=550,  type=int, help='[int] number of crossover samples')
    wr_parser.add_argument('--target', '-t', default=11_000, type=int, help='[int] number of samples in each batch index')

    gl_parser = subparsers.add_parser('griffinlim')
    mg_parser = subparsers.add_parser('melgan')

    args = parser.parse_args()
    
    print(args)

    assert args.vocoder in {'griffinlim', 'wavernn', 'melgan'}, \
        'Please provide a valid vocoder! Choices: [\'griffinlim\', \'wavernn\', \'melgan\']'

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        config = read_config(args.config)
        paths = Paths(config['data_path'], config['voc_model_id'], config['tts_model_id'])
        checkpoint_path = paths.taco_checkpoints / 'latest_weights.pyt'

    tts_model, config = load_taco(checkpoint_path)
    dsp = DSP.from_config(config)
    
    voc_checkpoint_path = args.voc_checkpoint
    if voc_checkpoint_path is None:
        config = read_config(args.config)
        paths = Paths(config['data_path'], config['voc_model_id'], config['tts_model_id'])
        voc_checkpoint_path = paths.voc_checkpoints / 'latest_weights.pyt'    

    voc_model, voc_dsp = None, None
    if args.vocoder == 'wavernn':
        voc_model, voc_config = load_wavernn(voc_checkpoint_path)
        voc_dsp = DSP.from_config(voc_config)

    out_path = Path('model_outputs')
    out_path.mkdir(parents=True, exist_ok=True)
    
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    tts_model.to(device)
    cleaner = Cleaner.from_config(config)
    tokenizer = Tokenizer()

    print('Using device:', device)
    if args.input_text:
        texts = [args.input_text]
    else:
        with open('sentences.txt', 'r', encoding='utf-8') as f:
            texts = f.readlines()

    tts_k = tts_model.get_step() // 1000

    if args.vocoder == 'griffinlim':
        simple_table([('Forward Tacotron', str(tts_k) + 'k'),
                      ('Vocoder Type', 'Griffin-Lim')])

    elif args.vocoder == 'melgan':
        simple_table([('Forward Tacotron', str(tts_k) + 'k'),
                      ('Vocoder Type', 'MelGAN')])

    # simple amplification of pitch
    pitch_function = lambda x: x * args.amp
    
    concatenated = np.zeros([0], dtype=np.single)

    for i, x in enumerate(texts, 1):
        print(f'\n| Generating {i}/{len(texts)}')
        x = cleaner(x)
        x = tokenizer(x)
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        wav_name = f'{i}_taco_{tts_k}k_{args.vocoder}'

        _, m, _ = tts_model.generate(x=x, steps=args.steps)
        if args.vocoder == 'melgan':
            m = torch.tensor(m).unsqueeze(0)
            torch.save(m, out_path / f'{wav_name}.mel')
        if args.vocoder == 'wavernn':
            m = torch.tensor(m).unsqueeze(0)
            wav = voc_model.generate(mels=m,
                                     batched=True,
                                     target=args.target,
                                     overlap=args.overlap,
                                     mu_law=voc_dsp.mu_law)
            dsp.save_wav(wav, out_path / f'{wav_name}.wav')
        elif args.vocoder == 'griffinlim':
            wav = dsp.griffinlim(m)
            dsp.save_wav(wav, out_path / f'{wav_name}.wav')            
        if args.vocoder != 'melgan':
            wav = np.append(wav, np.zeros(int(22050*0.25), dtype=np.single))
            concatenated = np.append(concatenated, wav)

    if len(texts) > 1:
        dsp.save_wav(concatenated, out_path / f'taco_{tts_k}k_{args.vocoder}-full.wav')

    print('\n\nDone.\n')
