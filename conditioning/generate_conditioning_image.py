import argparse
import os

from utils import build_canny_map, build_color_palette

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image file")
    parser.add_argument("--output_image", type=str, help="Path to the output image file")
    parser.add_argument("--target_condition", type=str, required=True, help="Target conditoning image type to generate.")
    args = parser.parse_args()
    if args.target_condition.lower() not in ["canny", "palette"]:
        raise ValueError("Please select 'canny' or 'palette' to be the conditioning images.")
    
    if args.target_condition == "canny":
        output_img = build_canny_map(args.input_image, src_byte=None, plot=False, return_byte=False)
    elif args.target_condition == "palette":
        output_img = build_color_palette(args.input_image, src_byte=None, plot=False, return_byte=False)
    os.makedirs(os.path.dirname(args.output_image), exist_ok=True)
    output_img.save(args.output_image)
