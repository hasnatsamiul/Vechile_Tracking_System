import SegmentCharacters
import pickle
import cv2
import csv
from datetime import datetime
 

print("Loading model")
filename = './finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

print('Model loaded. Predicting characters of number plate')
classification_result = []
for each_character in SegmentCharacters.characters:
    # converts it to a 1D array
    each_character = each_character.reshape(1, -1);
    result = model.predict(each_character)
    classification_result.append(result)

print('Classification result')
print(classification_result)

plate_string = ''
for eachPredict in classification_result:
    plate_string += eachPredict[0]

print('Predicted license plate')
print(plate_string)

# it's possible the characters are wrongly arranged
# since that's a possibility, the column_list will be
# used to sort the letters in the right order

column_list_copy = SegmentCharacters.column_list[:]
SegmentCharacters.column_list.sort()
rightplate_string = ''
for each in SegmentCharacters.column_list:
    rightplate_string += plate_string[column_list_copy.index(each)]

print('License plate')
print(rightplate_string)

now = datetime.now()
date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
print("date and time:",date_time)

# initialize .csv
Heading = ['platename','predictplate','date&time']

valx = [plate_string,rightplate_string,date_time ]

with open('traffic_measurement.csv', 'a') as csv_file:
    
    
    
    
    csv_writer = csv.writer(csv_file, delimiter=',')
    
    #csv_writer.writerow(Heading)
    
    csv_writer.writerow(valx)
#f = open('traffic_measurement.csv', 'a')

#with f:
    
    #fnames = ['platename', 'predictplate', 'date&time' ]
    #writer = csv.DictWriter(f, fieldnames=fnames)    

    #writer.writeheader()
    #writer.writerow( [plate_string] , [rightplate_string] ,[date_time] )


with open('dataset.csv') as csv_file:
     if plate_string in csv_file.read():
         print('Its a valid resgistered car')
     else:
        print('Its not a valid registered car')
