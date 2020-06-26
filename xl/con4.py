import pandas as pd
import os
#create an empty dataframe which will have all the combined data
mergedData = pd.DataFrame()
for files in os.listdir():
    #make sure you are only reading excel files
    if files.endswith('.xlsx'):
        data = pd.read_excel(files, index_col=None)
        mergedData = mergedData.append(data)
       
