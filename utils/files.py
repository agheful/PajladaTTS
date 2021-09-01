import pickle
import yaml
from pathlib import Path
from typing import Union, List, Any, Dict, Tuple


def get_files(path: Union[str, Path], extension='.wav') -> List[Path]:
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))


def pickle_binary(data: object, file: Union[str, Path]) -> None:
    with open(str(file), 'wb') as f:
        pickle.dump(data, f)


def unpickle_binary(file: Union[str, Path]) -> Any:
    with open(str(file), 'rb') as f:
        return pickle.load(f)


def read_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    with open(path, 'w+', encoding='utf-8') as stream:
        yaml.dump(config, stream, default_flow_style=False)


def parse_schedule(schedule: List[str]) -> List[Tuple]:
    out = []
    for line in schedule:
        split = line.split(',')
        if len(split) == 4:
            r, lr, step, bs = split
            out.append((int(r), float(lr), int(step), int(bs)))
        else:
            lr, step, bs = split
            out.append((float(lr), int(step), int(bs)))
    return out




if __name__ == '__main__':
    config = read_config('../config.yaml')
    print(config)