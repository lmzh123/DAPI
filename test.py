import os
import sys
import cv2
import pytesseract
import pandas as pd
from StringIO import StringIO
import numpy as np
import json

def get_info(df):
	date = df[df['text'].str.match('DATE:')].index
	invoice = df[df['text'].str.match('No:')].index
	total = df[df['text'].str.match('TOTAL')].index
	date += 1
	invoice += 1
	total += 1
	
	date = date.tolist()
	invoice = invoice.tolist()
	total = total.tolist()

	return [df.iloc[date,6].values[0], df.iloc[invoice,6].values[0], 
		df.iloc[total,6].values[0]]

def create_json(df, filename):
	'''
	This function creates JSON from pandas data frame.
	Args:
		df (pandas.df).
	'''
	data = []
	for index, row in df.iterrows():
		x = int(row['left'])
		y = int(row['top'])
		w = int(row['width'])
		h = int(row['height'])
		text = row['text']
		conf = row['conf']/100.0
		
		item = {"word":text}
		item['position'] = {"x": x, "y": y, "w": w, "h": h}
		item['confidence'] = conf
		data.append(item)

	with open(filename+'.json', 'w') as outfile:
		json.dump(data, outfile)


def draw_detections(df, img, filename):
	'''
	This function visualizes the tesseract detections using openCV.
	Args:
		df (pandas.df).
		Image (numpy.array).
	'''
	font = cv2.FONT_HERSHEY_SIMPLEX
	
	for index, row in df.iterrows():
		x = int(row['left'])
		y = int(row['top'])
		w = int(row['width'])
		h = int(row['height'])
		text = row['text']
		conf = row['conf']/100.0

		cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 1)
		cv2.putText(img,text + ' ' + str(conf),(x,y), font, 0.3,
			(255,0,255),1,cv2.LINE_AA)
	cv2.imwrite(filename+'_OCR.jpg', img)

def parse_data(data):
	'''
	This function parses Tesseract data to pandas data frame.
	Args:
		Data(string).

	Returns:
		Data(pandas.df).
	'''
	df = pd.read_csv(StringIO(data), sep='\t',
		usecols=["left", "top", "width", "height", "conf", "text"])
	df['text'].replace(' ', np.nan, inplace=True)
	df.dropna(inplace = True)
	df.reset_index(inplace = True)
	return df

def ocr(img):
	'''
	This function performs OCR using Tesseract.
	Args:
		Image(numpy.array).

	Returns:
		words(pandas.df): Words, positions and confidence.
	'''
	info = pytesseract.image_to_data(img, config='-l eng --oem 1 --psm 3')
	
	df = parse_data(info)
	return df

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Usage: test.py invoice1.jpg invoice2.jpg ... invoicen.jpg ')
		sys.exit(1)
	n = len(sys.argv)

	total_info = []
	for i in range(1,n):
		im_path = sys.argv[i]
		filename = os.path.basename(im_path)
		filename = os.path.splitext(filename)[0]
		# read image
		img = cv2.imread(im_path)
		# Perform OCR
		df = ocr(img)
		draw_detections(df, img, filename)
		# Create JSON file with the same name as the image
		create_json(df, filename)
		# Create CSV
		total_info.append(get_info(df))

	total_info_df = pd.DataFrame(total_info)
	total_info_df.columns = ['date', 'invoice_number', 'total']
	total_info_df.to_csv('results.csv', sep='\t', encoding='utf-8')


