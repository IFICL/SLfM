import argparse
import numpy as np

def init_args(return_parser=False): 
    parser = argparse.ArgumentParser(description="""Configure""")

    # basic configuration 
    parser.add_argument('--exp', type=str, default='test101', help='checkpoint folder')
    parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run (default: 90)')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts) (default: 0)')

    parser.add_argument('--weights_audio', default='', type=str, metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--weights_vision', default='', type=str, metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--weights_optim', default='', type=str, metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--weights', default='', type=str, metavar='PATH', help='path to checkpoint (default: None)')

    parser.add_argument('--save_step', default=1, type=int)
    parser.add_argument('--valid_step', default=1, type=int)
    parser.add_argument('--test_mode', default=False, action='store_true')
    parser.add_argument('--eval', default=False, action='store_true')

    parser.add_argument('--clean_up', default=False, action='store_true', help='clean up the dataset when exit safely')
    parser.add_argument('--overwrite_data', default=False, action='store_true', help='overwrite the dataset when if exist')

    parser.add_argument('--seed', type=int, default=None, required=False)

    # Dataloader parameter
    parser.add_argument('--max_sample', default=-1, type=int)
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', default=24, type=int)

    parser.add_argument('--online_render', default=False, action='store_true')
    parser.add_argument('--time_sync', default=False, action='store_true')
    parser.add_argument('--not_load_audio', default=False, action='store_true')

    parser.add_argument('--n_source', type=int, default=1, required=False)
    parser.add_argument('--n_view', type=int, default=2, required=False)
    parser.add_argument('--add_noise', default=False, action='store_true')
    
    parser.add_argument('--with_dominant_sound', default=False, action='store_true')
    parser.add_argument('--dominant_snr', type=int, default=None, required=False)
    parser.add_argument('--ssl_flag',default=False, action='store_true', help="enable for sound localization downstream or zero-shot tasks")

    parser.add_argument('--audiobase_path', type=str, default='data/AI-Habitat/data-split/FMA', required=False)
    parser.add_argument('--save_audio', default=False, action='store_true', help='use to save rendered audio')
    parser.add_argument('--cond_clip_length', type=float, default=None, help='length could be 0.51 * N')
    parser.add_argument('--indirect_ratio', type=float, default=None, required=False)


    
    # optimizer parameters
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--optim', type=str, default='Adam', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--schedule', type=str, default='cos', choices=['none', 'cos', 'step'], required=False)

    # Loss parameters
    parser.add_argument('--loss_type', type=str, default='L1', required=False)
    
    # network parameters
    parser.add_argument('--setting', type=str, default='', required=False)

    parser.add_argument('--vision_backbone', type=str, default='resnet18', required=False)
    parser.add_argument('--audio_backbone', type=str, default='resnet18', required=False)
    parser.add_argument('--imagenet_pretrain', default=False, action='store_true')
    parser.add_argument('--use_real_imag', default=False, action='store_true')
    parser.add_argument('--use_mag_phase', default=False, action='store_true')

    parser.add_argument('--unet_input_nc', type=int, default=2, required=False)
    parser.add_argument('--unet_output_nc', type=int, default=2, required=False)

    parser.add_argument('--no_vision', default=False, action='store_true')
    parser.add_argument('--no_cond_audio', default=False, action='store_true')

    parser.add_argument('--freeze_camera', default=False, action='store_true')
    parser.add_argument('--freeze_audio', default=False, action='store_true')
    parser.add_argument('--freeze_generative', default=False, action='store_true')

    parser.add_argument('--mono2binaural', default=False, action='store_true')
    parser.add_argument('--color_jitter', default=False, action='store_true')
    
    parser.add_argument('--generative_loss_ratio', default=1., type=float, help='weight of generative loss')

    parser.add_argument('--azimuth_loss_type', type=str, default='classification', choices=['classification', 'regression'], required=False)
    parser.add_argument('--use_gt_rotation', default=False, action='store_true')

    
    # Geometric Loss Parameter
    parser.add_argument('--add_geometric', default=False, action='store_true')
    parser.add_argument('--geometric_loss_ratio', default=0., type=float, help='weight of geometric loss')
    parser.add_argument('--binaural_loss_ratio', default=0., type=float, help='weight of binaural loss')
    parser.add_argument('--symmetric_loss_ratio', default=0., type=float, help='weight of symmetric loss')

    parser.add_argument('--finer_rotation', default=False, action='store_true')
    parser.add_argument('--filter_sound', default=False, action='store_true')
    parser.add_argument('--activation', type=str, default='tanh', choices=['tanh', 'clamp', 'sigmoid'], required=False)
    parser.add_argument('--sound_permutation', default=False, action='store_true')
    parser.add_argument('--inverse_camera', default=False, action='store_true')

    if return_parser:
        return parser

    # global args
    args = parser.parse_args()

    return args
