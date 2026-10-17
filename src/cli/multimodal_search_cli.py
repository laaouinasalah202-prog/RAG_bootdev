import argparse
from posix import lseek
from lib.multimodal_search import verify_image_embedding, MultimodelSearch

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # verify_image_embedding command
    verify_parser = subparsers.add_parser("verify_image_embedding")
    verify_parser.add_argument("image_path")
    
    args = parser.parse_args()
    
    if args.command == "verify_image_embedding":
        verify_image_embedding(args.image_path)


if __name__ == "__main__":
    main()
