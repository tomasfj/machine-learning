'''
Read Data:
> save data to list
> find max and min for each theta (except last theta)

Write Data:
> normalize each cell (except last theta)


data = [ [t0], [t1], [t2], [t3], [t4], [t5], [t6], [t7], [t8] ]
t0 - Mean of the integrated profile
t1 - Standard deviation of the integrated profile.
t2 - Excess kurtosis of the integrated profile.
t3 - Skewness of the integrated profile.
t4 - Mean of the DM-SNR curve.
t5 - Standard deviation of the DM-SNR curve.
t6 - Excess kurtosis of the DM-SNR curve.
t7 - Skewness of the DM-SNR curve.
t8 - Class 

'''

import csv

data = []
maxs_mins = [[0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]
filename_read = 'pulsar_stars.csv'

with open(filename_read) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        # skip header row
        if line_count == 0:
            line_count += 1
        # add the values of the first row as the default max and min
        elif line_count == 1:
            for i in range( len(row)-1 ):
                # add first value to max
                maxs_mins[i][0] = float( row[i] )
                # add first value to min
                maxs_mins[i][1] = float( row[i] )
            
            data.append(row)
            line_count += 1
        # check if any value is either greater than or less than the default values
        # if so, substitute them
        else:
            for i in range( len(row)-1 ):
                if( float(row[i]) > maxs_mins[i][0] ):
                    maxs_mins[i][0] = float(row[i])
                if( float(row[i]) < maxs_mins[i][1] ):
                    maxs_mins[i][1] = float(row[i])

            data.append(row)
            line_count += 1

print("Done reading " + str(line_count) + " lines")
'''
for r in range(len(maxs_mins)):
    print("--T" + str(r) + "--")
    print("Max = " + str(maxs_mins[r][0]) )
    print("Min = " + str(maxs_mins[r][1]))
    print("\n")
'''

# write data
filename_write = 'normal_pulsar_stars.csv'
with open(filename_write, mode='a') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')

    for row in data:
        for i in range(len(row)-1):
            row[i] = ( float( row[i] ) - maxs_mins[i][1] ) / ( maxs_mins[i][0] - maxs_mins[i][1] )

        csv_writer.writerow(row)
