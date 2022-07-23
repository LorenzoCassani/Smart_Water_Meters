# Modules imports
import os.path
from os import path
from time import time
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Directories creation
os.makedirs("Figures/differential_readings_per_meter", exist_ok=True)


# Dataframe creation
while True:
    try:
        filename = str(input("Insert the file path of the .xlsx file that you want to analyze: "))
        if path.exists(filename):
            break
        print("The file path is not valid.")
    except Exception as e:
        print(e)
print("Creating pandas dataframe from .xlsx file...")
df = pd.read_excel(filename)

dict_meters = df['nome_locale'].to_dict() # Creates dictionary from dataframe
set_meters = set(dict_meters.values()) # Turns dictionary into a set
list_meters = sorted(set_meters) # Turns set into alphabetically ordered list

# Create list of dates in a specified range
sdate = date(2021, 1, 1) # Start date (included)
edate = date(2022, 3, 1) # End date (not included)
dates = pd.date_range(sdate, edate-timedelta(days=1), freq='d')
dates = dates.strftime('%d/%m/%Y')
dates = dates.tolist()


# Statistical analysis
time0 = time()
f = open('results_per_meter.txt', 'w')
for meter in list_meters:
    df_locale = df[((df['nome_locale'] == meter))]
    df_locale = df_locale.to_numpy()
    df_locale_data = []
    df_locale_lettura_cumulativa = []
    df_locale_lettura_differenziale = []
    for i in range(len(df_locale)):
        df_locale_data.append(df_locale[i][3].strftime('%d/%m/%Y'))
        df_locale_lettura_cumulativa.append(df_locale[i][2])
    df_locale_lettura_differenziale.append(0)
    for i in range(len(df_locale) - 1):
        df_locale_lettura_differenziale.append(df_locale[i + 1][2]-df_locale[i][2])
    
    restart = True
    while restart:
        for i in range(1, len(df_locale_data)):
            restart = False
            if df_locale_data[i] == df_locale_data[i - 1]:
                df_locale_data.pop(i)
                df_locale_lettura_differenziale.pop(i)
                restart = True
                break
    for i in range(len(dates)):
        if dates[i] not in df_locale_data:
            df_locale_data.insert(i, dates[i])
            df_locale_lettura_differenziale.insert(i, 0)
    
    df_locale_data = np.array(df_locale_data)
    df_locale_lettura_cumulativa = np.array(df_locale_lettura_cumulativa)
    df_locale_lettura_differenziale = np.array(df_locale_lettura_differenziale)
    mean = np.mean(df_locale_lettura_differenziale)
    sigma = np.std(df_locale_lettura_differenziale)
    k = 3 # Number of stddevs used to define water consumption peaks
    limit=mean + k*sigma # Peaks detection limit
    peaks = []
    peaks_dates = []
    negatives = []
    negatives_dates = []
    for i in range(len(df_locale_lettura_differenziale)):
        if df_locale_lettura_differenziale[i] > limit:
            peaks.append(df_locale_lettura_differenziale[i])
            peaks_dates.append(df_locale_data[i])
        elif df_locale_lettura_differenziale[i] < 0:
            negatives.append(df_locale_lettura_differenziale[i])
            negatives_dates.append(df_locale_data[i])
    print("\nWater meter name: {}".format(meter))
    f.write("Water meter name: {}\n".format(meter))
    if len(peaks) != 0:
        print("WARNING: {} consumption peak/peaks detected.".format(len(peaks)))
        f.write("WARNING: {} consumption peak/peaks detected.\n".format(len(peaks)))
        for i in range(len(peaks)):
            print("{}) {} on {}".format(i+1, peaks[i], peaks_dates[i]))
            f.write("{}) {} on {}\n".format(i+1, peaks[i], peaks_dates[i]))
    else:
        print("No consumption peaks detected.")
        f.write("No consumption peaks detected.\n")
    if len(negatives) != 0:
        print("WARNING: {} negative/negatives detected.".format(len(negatives)))
        f.write("WARNING: {} negative/negatives detected.\n".format(len(negatives)))
        for i in range(len(negatives)):
            print("{}) {} on {}".format(i+1, negatives[i], negatives_dates[i]))
            f.write("{}) {} on {}\n".format(i+1, negatives[i], negatives_dates[i]))
    else:
        print("No negatives detected.")
        f.write("No negatives detected.\n")
    f.write("\n")
    df_locale_data = [datetime.strptime(x, "%d/%m/%Y") for x in df_locale_data]
    for i in range(len(peaks_dates)):
        peaks_dates[i] = datetime.strptime(peaks_dates[i], "%d/%m/%Y")
    for i in range(len(negatives_dates)):
        negatives_dates[i] = datetime.strptime(negatives_dates[i], "%d/%m/%Y")
    
    
    # Matplotlib plots
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-large')
    plt.rc('ytick', labelsize='x-large')
    # Plot letture giornaliere
    plt.figure(figsize = (16, 9))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.plot(df_locale_data, df_locale_lettura_differenziale, color='k', label='consumo giornaliero')
    plt.scatter(peaks_dates, peaks, color='r', marker='o', label='picco')
    plt.scatter(negatives_dates, negatives, color='b', marker='o', label ='negativo')
    plt.axhline(y=0, color='k')
    plt.axhline(y=mean, color='k', linestyle='dashed', label='valor medio')
    plt.axhline(y=limit, color='r', linestyle='dashed', label='valore limite')
    plt.title("Consumi giornalieri {}".format(meter), fontsize='xx-large')
    plt.xlabel('data', fontsize='xx-large')
    plt.ylabel('consumo giornaliero', fontsize='xx-large')
    plt.gcf().autofmt_xdate()
    plt.grid()
    plt.legend(fontsize='large')
    plt.savefig('Figures/differential_readings_per_meter/{}.png'.format(meter), dpi=300)


print("\nExecution time (in minutes):", (time()-time0) / 60)
f.close() # Closes .txt file