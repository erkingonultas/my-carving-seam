from my_seam_carver import SeamCarver
import os 

def image_resize_without_mask(filename_input, filename_output, new_height, new_width):
    obj = SeamCarver(filename_input, new_height, new_width)
    obj.save_result(filename_output)


def image_resize_with_mask(filename_input, filename_output, new_height, new_width, filename_mask):
    obj = SeamCarver(filename_input, new_height, new_width, protect_mask=filename_mask)
    obj.save_result(filename_output)


def object_removal(filename_input, filename_output, filename_mask):
    obj = SeamCarver(filename_input, 0, 0, object_mask=filename_mask)
    obj.save_result(filename_output)

if __name__ == '__main__':
    """
    Put your image files in the input/images folder
    Put your mask files in the input/masks folder
    Ouput image will be saved to output/images folder with filename_output
    """
    folder_in = 'input'
    folder_out = 'output'

    filename_input = 'Broadway_tower_edit.jpg'
    filename_mask = 'mask.jpg'
    new_height = 2*968 / 3
    new_width = 2*1428 / 3

    input_image = os.path.join(folder_in, filename_input)
    input_mask = os.path.join(folder_in, filename_mask)
    output_image = os.path.join(folder_out, "result_"+filename_input)

    print("Processing...")

    image_resize_without_mask(input_image, output_image, new_height, new_width)
    # image_resize_with_mask(input_image, output_image, new_height, new_width, input_mask)
    # object_removal(input_image, output_image, input_mask)