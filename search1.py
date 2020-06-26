import cv2
import csv

searchfile=open('testout1.csv','r')
reader=csv.reader(searchfile,delimiter=',')
a=input('Enter the vehicle number:\n')
for row in reader:
    if a in row[0] or a in row[1] or a in row[2] or a in row[3]:
        print(row)
