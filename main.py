import argparse
import sys

import converter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input_dir',
        required=True,
        type=str,
        help="Please set the path for input directory."
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        required=True,
        type=str,
        help="Please set the path for the ouput directory.",
    )
    parser.add_argument(
        '-f',
        '--format',
        type=str,
        default="COCO",
        help="COCO for coco format. YOLO for YOLO format",
    )
    parser.add_argument(        
        '--train_rate',
        type=float,
        nargs="?",
        default=0,
        help="Please input the validation dataset size, for example 0.1 ",
    )    

    args = parser.parse_args(sys.argv[1:])
    
    if args.format == "COCO":
        labelme2coco = converter.labelme2coco()
        
        labelme2coco.convert(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            train_split_rate=args.train_rate,
        )
    elif args.format == "YOLO":
        labelme2yolo = converter.labelme2yolo()
        
        labelme2yolo.convert(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            train_split_rate=args.train_rate,
            copy_images=True
        )
    else:
        print("[ERROR] Check which format is available")
