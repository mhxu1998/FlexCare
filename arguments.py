import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='arguments')

    parser.add_argument('--data_path', type=str, help='Path to the data',
                        default='data')
    parser.add_argument('--ehr_path', type=str, help='Path to the ehr data',
                        default='data/ehr')
    parser.add_argument('--cxr_path', type=str, help='Path to the cxr data',
                        default='../data/mimic-cxr-jpg')
    parser.add_argument('--save_dir', type=str, help='Directory relative which all output files are stored',
                    default='checkpoints')


    parser.add_argument('--task', type=str, default='phenotyping', help='train or eval for in-hospital-mortality or phenotyping, decompensation, length-of-stay')
    parser.add_argument('--epochs', type=int, default=100, help='number of chunks to train')
    parser.add_argument('--device', type=str, default="cpu", help='cuda:number or cpu')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers for dataloader')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--model', type=str, default='our', help='our model or baselines')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)

    # args = argParser.parse_args()
    return parser
