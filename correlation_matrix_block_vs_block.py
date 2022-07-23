# Modules imports
import os.path
from os import path
from time import time
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Directories creation
os.makedirs("Figures/correlation_matrices/correlation_matrix_block_vs_block", exist_ok=True)


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

dict_blocks = df['nome_isola'].to_dict() # Creates dictionary from dataframe
set_blocks = set(dict_blocks.values()) # Turns dictionary into a set
list_blocks = sorted(set_blocks) # Turns set into alphabetically ordered list
dict_corr = {}


# Statistical analysis
time0 = time()
for block in list_blocks:

    df_isola = df[((df['nome_isola'] == block))]
    
    # Create list of dates in a specified range
    sdate = date(2021, 1, 1) # Start date (included)
    edate = date(2022, 3, 1) # End date (not included)
    dates = pd.date_range(sdate, edate-timedelta(days=1), freq='d')
    dates = dates.strftime('%d/%m/%Y')
    dates = dates.tolist()
    
    dict_meters = df_isola['nome_locale'].to_dict() # Creates dictionary from dataframe
    set_meters = set(dict_meters.values()) # Turns dictionary into a set
    list_meters = sorted(set_meters) # Turns set into alphabetically ordered list
    
    isola_lettura_differenziale = np.zeros(len(dates))
    
    for meter in list_meters:
        df_locale = df[((df['nome_locale'] == meter))]
        df_locale = df_locale.to_numpy()
        df_locale_data = []
        df_locale_lettura_differenziale = []
        for i in range(len(df_locale)):
            df_locale_data.append(df_locale[i][3].strftime('%d/%m/%Y'))
        df_locale_lettura_differenziale.append(0)
        for i in range(len(df_locale) - 1):
            df_locale_lettura_differenziale.append(df_locale[i + 1][2] - df_locale[i][2])
            
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
        
        df_locale_lettura_differenziale = np.array(df_locale_lettura_differenziale)
        isola_lettura_differenziale = isola_lettura_differenziale + df_locale_lettura_differenziale
        
    dict_corr.update({block: isola_lettura_differenziale})

df_corr = pd.DataFrame(data=dict_corr)
corr_matrix = df_corr.corr(method='kendall')

# Matplotlib & seaborn plots
sns.set(rc={'figure.figsize': (16, 9)})
sns.set_style(style='white')
# Seaborn heatmap correlation matrix
plt.figure(figsize = (16, 9))
plt.title("Matrice delle correlazioni isola-isola (Kendall)", fontsize='xx-large')
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, annot_kws={"size": 8}, vmax=1, vmin=-1, center=0, cmap='vlag', mask=mask)
plt.savefig('Figures/correlation_matrices/correlation_matrix_block_vs_block/correlation_matrix_block_vs_block.png', dpi=300)
plt.show()
print("\nExecution time (in minutes):", (time()-time0) / 60)