import argparse
from lib.multimodal_search import image_search_command
from lib.multimodal_search import verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Generate an embedding for an image and print its shape"
    )
    verify_parser.add_argument("image_path", type=str, help="Path to the image file")
    image_search_parser = subparsers.add_parser("image_search")
    image_search_parser.add_argument("img_path")
    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)

        case "image_search":
            result = image_search_command(args.img_path)
            for n, r in enumerate(result, 1):
                print(f'''{n}. {r["title"]} (similarity: {r["cosine_sim"]:.1f}
                    {r["description"]}'''
                )
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()