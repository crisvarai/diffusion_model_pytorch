import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./")
    parser.add_argument('--imgsize', type=int, default=64)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--timesteps', type=int, default=300)
    parser.add_argument('--wgts_path', type=str, default="./weights/best_model_py.pth")
    args = parser.parse_args()
    return args