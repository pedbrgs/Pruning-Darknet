import os
import cv2
import argparse

def yolo2bbox(dim, coord_norm):

    """ Converts normalized coordinates in YOLO format to [xmin, ymin, xmax, ymax] format. """

    # xmin = w_image * (xmin_norm - xmax_norm/2)
    xmin = dim[0] * (coord_norm[0] - coord_norm[2]/2)
    # xmax = w_image * (xmax_norm/2 + xmin_norm)
    xmax = dim[0] * (coord_norm[2]/2 + coord_norm[0])
    # ymin = h_image * (ymin_norm - ymax_norm/2)
    ymin = dim[1] * (coord_norm[1] - coord_norm[3]/2)
    # ymax = h_image * (ymax_norm/2 + ymin_norm)
    ymax = dim[1] * (coord_norm[3]/2 + coord_norm[1])

    return int(xmin), int(ymin), int(xmax), int(ymax)

parser = argparse.ArgumentParser()
parser.add_argument('--output', type = str, help = 'Output path.')
parser.add_argument('--input', type = str, help = 'Input path.')
parser.add_argument('--image', type = str, help = 'Image path.')
opt = parser.parse_args()

for subdir, dirs, files in os.walk(opt.input):

    for filename in sorted(files):

        if filename.endswith('.txt'):

            input_path = opt.input + filename
            output_path = opt.output + filename
            image_path = opt.image + filename.split('.')[0] + '.jpg'
            
            input = open(input_path)
            output = open(output_path, 'w')
            image = cv2.imread(image_path)

            print(image_path)
            height, width, channels = image.shape
            count = 0

            for line in input.readlines():
                count += 1
            output.write('%d\n' % (count))
            input.seek(0)

            for line in input.readlines():
                
                class_id = int(line.split(' ')[0])
                if class_id == 0:
                    class_name = 'smoke'
                else:
                    class_name = 'fire'
                xmin = float(line.split(' ')[1])
                ymin = float(line.split(' ')[2])
                xmax = float(line.split(' ')[3])
                ymax = float(line.split(' ')[4])

                xmin, ymin, xmax, ymax = yolo2bbox((width, height), (xmin, ymin, xmax, ymax))

                output.write('%d %d %d %d %s\n' % (int(xmin), int(ymin), int(xmax), int(ymax), class_name))


            output.close()
