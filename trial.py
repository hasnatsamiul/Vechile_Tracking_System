import csv
files={'Jutraffic.csv','malibagtraffic.csv','traffic_measurement.csv'}
for file in files:
    with open(file,'r') as f1:
        csv_reader = csv.reader(f1,delimiter=',')
        with open('testout1.csv','a',newline='') as f2:
            csv_writer = csv.writer(f2,delimiter=',')
            for row in csv_reader:
                csv_writer.writerow(row)
        
