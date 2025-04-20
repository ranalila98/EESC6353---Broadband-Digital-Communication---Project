import numpy as np
from scipy.fft import fft, fftshift
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="CSI Preprocessing Pipeline")
    parser.add_argument('--data_folder', type=str, default='data', help='Input folder with raw CSI .npy files')
    parser.add_argument('--output_folder', type=str, default='preprocessed_data', help='Output folder for processed .npy files')
    parser.add_argument('--Bt', type=int, default=128, help='Time batch size')
    parser.add_argument('--Nf', type=int, default=15, help='Reduced Number of subcarriers')
    parser.add_argument('--Nsc', type=int, default=114, help='Original number of subcarriers')
    parser.add_argument('--Btw', type=int, default=14, help='Window width to crop from Bt')
    parser.add_argument('--S', type=int, default=100, help='Sample Frequency- Fixed')
    return parser.parse_args()


def load_data(file_path):
    return np.load(file_path)

def group_into_batches(data, Bt):
    Nt = data.shape[0] // Bt
    return data[:Nt * Bt].reshape(Nt, Bt, *data.shape[1:])

def reduce_subcarriers(data, Nf, Nsc):
    idx = np.linspace(0, Nsc-1, Nf, dtype=int)
    return data[:, :, :, :, idx]

def combine_antenna_dimensions(data):
    return data.reshape(data.shape[0], data.shape[1], -1, data.shape[4])

def discard_phase(data):
    return np.abs(data)

def normalize_data(data, epsilon=1e-10):
    return data / (data[:, 0:1, :, :] + epsilon)

def apply_2d_dft(data):
    return np.abs(fftshift(fft(data, axis=1), axes=(1, 3)))

def log_transform(data):
    return np.log10(1 + data)

def crop_data(data, Bt, Btw):
    start = (Bt - Btw) // 2
    return data[:, start:start + Btw, :, :]

def preprocess_csi_data(file_path, Bt, Nf, Nsc, Btw):
    data = load_data(file_path)
    data_batched = group_into_batches(data, Bt)
    data_reduced = reduce_subcarriers(data_batched, Nf, Nsc)
    data_combined = combine_antenna_dimensions(data_reduced)
    data_magnitude = discard_phase(data_combined)
    data_normalized = normalize_data(data_magnitude)
    data_dft = apply_2d_dft(data_normalized)
    data_log = log_transform(data_dft)
    return crop_data(data_log, Bt, Btw)


def main():
    args = parse_args()

    data_folder = Path(args.data_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    file_suffixes = [
        'a_1_empty', 'a_1_occupied',
        'a_36_empty', 'a_36_occupied',
        'b_1_empty', 'b_1_occupied',
        'b_36_empty', 'b_36_occupied',
        'c_1_empty', 'c_1_occupied',
        'c_36_empty', 'c_36_occupied'
    ]

    for suffix in file_suffixes:
        try:
            input_path = data_folder / f'63000x_{suffix}.npy'
            preprocessed = preprocess_csi_data(input_path, args.Bt, args.Nf, args.Nsc, args.Btw)

            channel = '1' if '_1_' in suffix else '36'
            save_dir = output_folder / f'channel_{channel}'
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path = save_dir / f'preprocessed_{suffix}.npy'
            np.save(save_path, preprocessed)

            print(f"Saved: {save_path}")

        except Exception as e:
            print(f"Error processing {suffix}: {e}")

    print("Done.")

if __name__ == '__main__':
    main()
