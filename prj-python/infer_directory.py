import cv2
import os
from centerface import CenterFace
import argparse

parser = argparse.ArgumentParser(description='Directory Inference')
parser.add_argument('-i', '--input', type=str, help='Input dir', required=True)
parser.add_argument('-o', '--output', type=str,
					help='Output/ results dir', required=True)

args = parser.parse_args()


def test_dir():
	input_path = args.input
	save_path = args.output
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	landmarks = True
	#centerface = CenterFace(landmarks=landmarks)

	for filename in os.listdir(input_path):
		if filename.endswith(".jpg"):
			print("Processing File: ", filename)
			name = os.path.splitext(filename)[0]

		# print(save_path + im_dir)
		img = cv2.imread(os.path.join(input_path, filename))
		if isinstance(img, type(None)):
			continue
		h, w = img.shape[:2]

		centerface = CenterFace(landmarks=landmarks)
		if landmarks:
			dets, lms = centerface(img, h, w, threshold=0.35)
		else:
			dets = centerface(img, threshold=0.35)

		f = open(save_path + '/' + name + '.txt', 'w')
		# f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, filename)))
		f.write('{:d}\n'.format(len(dets)))
		for b in dets:
			x1, y1, x2, y2, score = b
			f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), score))

			cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (2, 255, 0), 1)

			# if landmarks:
			# 	for lm in lms:
			# 		for i in range(0, 5):
			# 			cv2.circle(img, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
		f.close()
		cv2.imwrite(os.path.join(save_path, name + '_result.jpg'), img)


if __name__ == '__main__':

	test_dir()
