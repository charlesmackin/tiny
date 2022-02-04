import argparse

def train_parser():
    parser = argparse.ArgumentParser(description='Process training inputs...')

    parser.add_argument('--model_dir',
                        metavar=str,
                        default='/u/cmackin/tiny-master/v0.5/training/anomaly_detection/ares_hwa_models',
                        help='model chkpt directory')

    parser.add_argument('--model_name',
                        metavar=str,
                        default='chkpt_0.pt',
                        help='model checkpoint name')

    parser.add_argument('--dev_dir',
                        metavar=str,
                        default='/u/cmackin/tiny-master/v0.5/training/anomaly_detection/dev_data',
                        help='dev_data directory location')

    parser.add_argument('--eval_dir',
                        metavar=str,
                        default='/u/cmackin/tiny-master/v0.5/training/anomaly_detection/ares_eval_data',
                        help='eval_data directory location')

    parser.add_argument('--result_dir',
                        metavar=str,
                        default='/u/cmackin/tiny-master/v0.5/training/anomaly_detection/ares_hwa_results',
                        help='test results .csv directory')

    parser.add_argument('--bias', type=int, default=1, help='turn bias on or off')

    parser.add_argument('--max_fpr', type=float, default=0.1)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--frames', type=int, default=5)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--win_length', type=int, default=None)
    parser.add_argument('--hop_length', type=int, default=512)
    parser.add_argument('--power', type=int, default=2)
    parser.add_argument('--log_mel_start_ind', type=int, default=50)
    parser.add_argument('--log_mel_stop_ind', type=int, default=250)
    parser.add_argument('--h_size', type=int, default=128)
    parser.add_argument('--c_size', type=int, default=24)
    parser.add_argument('--l2_reg', type=float, default=0.0)
    parser.add_argument('--dropout_ratio', type=float, default=0.0)
    parser.add_argument('--prune', type=float, default=0.0)
    parser.add_argument('--resume_from_checkpoint', type=bool, default=False)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--loss', type=str, default="MSE")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--validation_split', type=float, default=0.1)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--max_weight', type=float, default=2.)
    parser.add_argument('--max_grad_norm', type=float, default=10000.)
    parser.add_argument('--max_z', type=float, default=1000000)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--noise_type', type=str, default='custom-log-normal')
    parser.add_argument('--noise_std', type=float, default=0.0)
    parser.add_argument('--out_noise', type=float, default=0.0)
    parser.add_argument('--chekpoint_frequency', type=int, default=1)

    parser.add_argument('--version', action='store_true', help="show application version")
    parser.add_argument('--eval', type=bool, default=False, help="run mode Evaluation")
    parser.add_argument('--dev', type=bool, default=True, help="run mode Development")

    return parser

def test_parser():
    parser = argparse.ArgumentParser(description='Process test inputs...')

    parser.add_argument('--model_dir',
                        metavar=str,
                        default='/u/cmackin/tiny-master/v0.5/training/anomaly_detection/ares_hwa_models',
                        help='model chkpt directory')

    parser.add_argument('--model_name',
                        metavar=str,
                        default='chkpt_0.pt',
                        help='model checkpoint name')

    parser.add_argument('--dev_dir',
                        metavar=str,
                        default='/u/cmackin/tiny-master/v0.5/training/anomaly_detection/dev_data',
                        help='dev_data directory location')

    parser.add_argument('--eval_dir',
                        metavar=str,
                        default='/u/cmackin/tiny-master/v0.5/training/anomaly_detection/ares_eval_data',
                        help='eval_data directory location')

    parser.add_argument('--result_dir',
                        metavar=str,
                        default='/u/cmackin/tiny-master/v0.5/training/anomaly_detection/ares_hwa_results',
                        help='test results .csv directory')

    parser.add_argument('--result_file',
                        metavar=str,
                        default='ares_hwa_result.csv',
                        help='test results .csv directory')

    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--version', action='store_true', help="show application version")
    parser.add_argument('--eval', type=bool, default=False, help="run mode Evaluation")
    parser.add_argument('--dev', type=bool, default=True, help="run mode Development")

    return parser