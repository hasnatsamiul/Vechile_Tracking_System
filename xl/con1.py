import pandas as pd
import cvs
import os

# filenames
excel_names = ["Jutraffic.cvs", "malibag.cvs", "traffic_measurement"]

# read them in
cvs = [pd.cvsFile(name) for name in excel_names]

# turn them into dataframes
frames = [x.parse(x.sheet_names[0], header=None,index_col=None) for x in excels]

# delete the first row for all frames except the first
# i.e. remove the header row -- assumes it's the first
frames[1:] = [df[1:] for df in frames[1:]]

# concatenate them..
combined = pd.concat(frames)

# write it out
combined.to_excel("c.cvs", header=False, index=False)

