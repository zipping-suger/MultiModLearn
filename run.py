import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Run a specific method script')
    parser.add_argument('--method', type=str, required=False, default='ibc', choices=['mlp', 'cgan', 'ebgan','cwgan','cvae', 'reinforce', 'ibc', 'direct_supervise'], help='Method to run')
    parser.add_argument('--data_file_path', type=str, required=False, default='data/gradient_data_rs.npy', help='Path to the data file')
    args = parser.parse_args()

    method_script = f"methods/{args.method}.py"
    command = ['python', method_script]

    if args.data_file_path:
        command.extend(['--data_file_path', args.data_file_path])

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the script: {e}")

if __name__ == "__main__":
    main()