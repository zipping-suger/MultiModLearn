import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Run a specific method script')
    parser.add_argument('--method', type=str, required=True, choices=['mlp', 'cgan', 'cvae', 'reinforce', 'ibc'], help='Method to run')
    args = parser.parse_args()

    method_script = f"methods/{args.method}.py"

    try:
        subprocess.run(['python', method_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the script: {e}")

if __name__ == "__main__":
    main()