import argparse

# parser.add_argument("--[arg_name]", required=[required_bool], type=[arg_type], default=[deault], help=[help string])

def parsing(mode="args"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir", type=str, default="/root/dev/deepul/exp")
    parser.add_argument("--data_dir", type=str, default="/root/dev/deepul/homeworks/hw2/data")
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_epoch_step", type=int, default=10)
    
    if mode == "args":
        args = parser.parse_args()
        return args
    else:
        return parser