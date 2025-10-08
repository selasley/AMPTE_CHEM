sum_csv_files.app is a small application created by py2app that will sum the values in 
2-D histogram csv files from AMPTE Web App.  There are three modes of operation

1) Drop two or more csv files onto the app icon.  The app will create a file named
sum.csv containing the sum of the 2-D histogram values in the directory of the first 
file dropped.

2) Double click on the app icon.  This will sum the values in all .csv files in the same
folder as sum_csv_files.app and save the result to a file named sum.csv in the same folder
as the app.

3) If you have python3.11 installed you can run sum_csv_files from the command line with
a command such as
path_to_sum_csv_files.app/Contents/MacOS/sum_csv_files  path_to_csv_file1 path_to_csv_file2 path_to_csv_file3
For example, if you are in the same directory as sum_csv_files.app in Terminal.app you 
could use the command
./sum_csv_files.app/Contents/MacOS/sum_csv_files ~/Desktop/csv_files/*.csv
to sum all .csv files in the csv_files folder on your Desktop


In all cases the three header lines in sum.csv file will come from the last .csv file
processed.  The column and row indices will come from the first .csv file processed.  All 
csv files must have the same dimensions (number of rows and number of columns) and must 
have three header lines above the histogram data