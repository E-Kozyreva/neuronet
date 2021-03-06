import argparse
import numpy as np
import cv2
import sys


def make_cat_passport_image(input_image_path, haar_model_path):

    image = cv2.imread("cat.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
    normalizedImg = np.zeros((800, 800))
    gray = cv2.normalize(gray, normalizedImg, 0, 255, cv2.NORM_MINMAX)

    detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))

    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.imwrite('out.jpg', image)

    cv2.imshow("CAT", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def build_argparser():
    parser = argparse.ArgumentParser(
        description='Speech denoising demo', add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model', type=str, required=True,
                      help='Required. Path to .XML file with pre-trained model.')
    args.add_argument('-i', '--input', type=str, required=True,
                      help='Required. Path to input image')
    return parser


def main():
    
    args = build_argparser().parse_args()
    make_cat_passport_image(args.input, args.model)

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
