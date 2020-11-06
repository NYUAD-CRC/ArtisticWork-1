#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  14 58:47:53 2020

@author: Karunakar
"""

import csv
import os
import pandas as pd
import numpy as np


#if __name__ == '__main__':
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Change the directory to the directory where you would want to save the csv files
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

directory = 'C://Users/krm7/Dropbox/009 Art and Data with Karun/001 Contemporary Living Artists'

os.chdir(directory)


###################################################################################################

#################################################
#                                               #
#           Starting the analysis               #
#                                               #
#################################################
# Reading artist level Master_id_year_panel_nationality_withoutgap_KL_similarity_count.dta

Df_ArtMas = pd.read_stata('./artist level Master_id_year_panel_nationality_withoutgap_KL_similarity_count.dta')
print(Df_ArtMas.describe())
Df_ArtMas.to_csv('artist level Master_id_year_panel_nationality_withoutgap_KL_similarity_count.csv',encoding='utf-8',index=False)



#####################################################################################################################

# Reading Artist level Panel_nationality_withoutgap_similarity_hofstede_2018 Master.dta

Df_ArtPan = pd.read_stata('Artist level Panel_nationality_withoutgap_similarity_hofstede_2018 Master.dta')
print(Df_ArtPan.describe())
Df_ArtPan.to_csv('Artist level Panel_nationality_withoutgap_similarity_hofstede_2018 Master.csv',encoding='utf-8',index=False)

#####################################################################################################################
#Reading artwork level_transaction_2D_only_artist attribute.dta

Df_ArtLevel = pd.read_stata('artwork level_transaction_2D_only_artist attribute.dta')
print(Df_ArtLevel.describe())
Df_ArtLevel.to_csv('artwork level_transaction_2D_only_artist attribute.csv',encoding='utf-8',index=False)

#####################################################################################################################

#Reading df_for_ml_stata.dta

Df_mldata = pd.read_stata('df_for_ml_stata.dta')
print(Df_mldata.describe())
Df_mldata.to_csv('df_for_ml_stata.csv',encoding='utf-8',index=False)

#################################################################
# Dropping duplicate rows and null value rows from the merge keys
Df_ArtMas = Df_ArtMas.drop_duplicates()
Df_ArtMas.dropna(subset=["alive_id", "sales_id", "year"],inplace=True)
print(Df_ArtMas.head())
Df_ArtPan = Df_ArtPan.drop_duplicates()
Df_ArtPan.dropna(subset=["alive_id", "sales_id", "year"],inplace=True)
Df_ArtLevel = Df_ArtLevel.drop_duplicates()
Df_ArtLevel.dropna(subset=["alive_id", "sales_id", "year","case_id"],inplace=True)
Df_mldata = Df_mldata.drop_duplicates()
Df_mldata.dropna(subset=["case_id","year"],inplace=True)

#Df_mlfinal = pd.merge(Df_ArtMas,Df_ArtPan, on=("alive_id", "sales_id","year"), how='inner')

# Merging the files for final ML output and dropping the duplicates columns

Df_mlfinal = Df_ArtMas.merge(Df_ArtPan, on=("alive_id", "sales_id","year"),
             how='inner', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

##Df_mlfinal.to_csv('Df_mlfinal.csv',encoding='utf-8',index=False)

Df_mlfinout = Df_mlfinal.merge(Df_ArtLevel, on=("alive_id", "sales_id","year"),
             how='inner', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
Df_mlfinout = Df_mlfinout.replace(np.nan,'0')
Df_mlfinout = Df_mlfinout.replace(' ','0')
#cols_to_use = Df_mldata.columns.difference(Df_mlfinout.columns)
#Df_mlfinout = Df_mlfinout[cols_to_use]
##Df_mlfinout.to_csv('Df_mlfinout.csv',encoding='utf-8',index=False)

Df_mlout = Df_mlfinout.merge(Df_mldata, on=("case_id","year"),
             how='inner', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
caseid = list(Df_mlout['case_id'])
year = list(Df_mlout['year'])

Df_mloutfull = Df_mlfinout.merge(Df_mldata, on=("case_id","year"), indicator='Ind',
             how='outer', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
'''
for index, row in Df_mloutfull.iterrows():
    #if row['case_id'] in caseid and row['year'] in year:
    if row['case_id'] in caseid :
        Df_mloutfull['lkp_inp4'] = '1'
    else:
        Df_mloutfull['lkp_inp4'] = '0'
'''

##Df_mlout.to_csv('Df_mlout.csv',encoding='utf-8',index=False)
Df_mloutfull.to_csv('Df_mloutfull.csv',encoding='utf-8',index=False)

cols = Df_mldata.columns

##Df_mlout[cols].to_csv('Df_mlout1.csv',encoding='utf-8',index=False)
cols_to_use = Df_mlfinout.columns.difference(Df_mldata.columns)
print(cols_to_use)
df = pd.DataFrame(cols_to_use,columns=['Column_Name'])
df.to_csv('cols_to_use.csv',encoding='utf-8',index=False)
cols_to_use1 = Df_mldata.columns.difference(Df_mlfinout.columns)
print(cols_to_use1)
df1 = pd.DataFrame(cols_to_use1,columns=['Column_Name'])
df1.to_csv('cols_to_use1.csv',encoding='utf-8',index=False)

