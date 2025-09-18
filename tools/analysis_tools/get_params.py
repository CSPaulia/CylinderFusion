# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from mmengine import Config, DictAction
from mmengine.registry import init_default_scope

from mmdet3d.registry import MODELS

from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def params_to_string(num_params: float,
                     units: Optional[str] = None,
                     precision: int = 2) -> str:
    """Convert parameter number into a string.

    Args:
        num_params (float): Parameter number to be converted.
        units (str | None): Converted FLOPs units. Options are None, 'M',
            'K' and ''. If set to None, it will automatically choose the most
            suitable unit for Parameter number. Default: None.
        precision (int): Digit number after the decimal point. Default: 2.

    Returns:
        str: The converted parameter number with units.

    Examples:
        >>> params_to_string(1e9)
        '1000.0 M'
        >>> params_to_string(2e5)
        '200.0 k'
        >>> params_to_string(3e-9)
        '3e-09'
    """
    if units is None:
        if num_params // 10**6 > 0:
            return str(round(num_params / 10**6, precision)) + ' M'
        elif num_params // 10**3:
            return str(round(num_params / 10**3, precision)) + ' k'
        else:
            return str(num_params)
    else:
        if units == 'M':
            return str(round(num_params / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(num_params / 10.**3, precision)) + ' ' + units
        else:
            return str(num_params)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    model = MODELS.build(cfg.model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = params_to_string(num_params)
    split_line = '=' * 30
    print(f'Params: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
