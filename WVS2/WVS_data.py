#!/usr/bin/env python
# coding: utf-8

# In[2403]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
import matplotlib.colors as colors
from ast import literal_eval
from collections import Counter
import matplotlib.font_manager as fm

import plotly.graph_objs as go

from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import dendrogram, to_tree

from scipy.spatial.distance import pdist, squareform
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import KNNImputer
from sklearn.impute import KNNImputer
import warnings
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")


# In[2404]:


#import data from wave 7 in zipped format
df = pd.read_csv("WVS_Cross-National_Wave_7_csv_v5_0.csv.gz",  compression='gzip', encoding= 'utf8')
df.head()


# In[2405]:


#check shape and columns
print (df.columns)
df.shape


# In[2406]:


df.info()


# In[2440]:


#                0           1                   2                3             4           5                 6
#"iso3_code": ["#index", "Country/Territory", "iso3_code", "language_family", "Sample", "Fieldwork_period", "Mode", "Languages_fielded"],

dict_countrycode2info={
"AND": ["1", "Andorra", "AND", "Latin", "1004", "01-06-2018-22-09-2018", "PAPI", "Catalan,English,Spanish,French"],
"ARG": ["2", "Argentina", "ARG", "Latin", "1003", "04-07-2017-19-07-2017", "PAPI", "Spanish"],
"ARM": ["4", "Armenia", "ARM", "Isolate", "1223", "07-05-2021-07-06-2021", "CAPI", "Armenian"],
"AUS": ["3", "Australia", "AUS", "Anglosphere", "1813", "06-04-2018-06-08-2018", "Mail/Post", "English"],
"BGD": ["5", "Bangladesh", "BGD", "Indo-Iranian", "1200", "03-12-2018-24-12-2018", "PAPI", "Bengali"],
"BOL": ["6", "Bolivia", "BOL", "Latin", "2067", "18-01-2017-07-03-2017", "CAPI", "Spanish"],
"BRA": ["7", "Brazil", "BRA", "Latin", "1762", "15-05-2018-11-06-2018", "CAPI", "Portuguese"],
"CAN": ["8", "Canada", "CAN", "Anglosphere", "4018", "02-10-2020-19-10-2020", "CAWI", "English,French"],
"CDE": ["65", "CDE", "CDE", "Anglosphere", "0", "11-02-2020-23-03-2020", "CAPI", "English"],
"CDF": ["66", "CDF", "CDF", "Latin", "0", "11-02-2020-23-03-2020", "CAPI", "French"],
"CHL": ["9", "Chile", "CHL", "Latin", "1000", "06-01-2018-05-02-2018", "CAPI", "Spanish"],
"CHN": ["10", "China", "CHN", "EastAsia", "3036", "07-07-2018-12-10-2018", "PAPI", "Chinese"],
"COL": ["11", "Colombia", "COL", "Latin", "1520", "30-11-2018-22-12-2018", "CAPI", "Spanish"],
"CYP": ["12", "Cyprus", "CYP", "Isolate", "1000", "13-05-2019-04-06-2019", "PAPI", "Greek,Turkish"],
"CZE": ["13", "Czechia", "CZE", "Slavic", "1200", "11-02-2022-13-05-2022", "CAPI", "Czech"],
"DEU": ["17", "Germany", "DEU", "Germanic", "1528", "25-10-2017-31-03-2018", "CAPI", "German"],
"ECU": ["14", "Ecuador", "ECU", "Latin", "1200", "24-01-2018-03-03-2018", "CAPI", "Spanish"],
"EGY": ["15", "Egypt", "EGY", "Semetic", "1200", "22-06-2018-07-07-2018", "CAPI", "Arabic"],
"ETH": ["16", "Ethiopia", "ETH", "Semetic", "1230", "06-02-2020-19-03-2020", "CAPI", "Amharic,Oromo,Tigris"],
"GBR": ["19", "Great_Britain", "GBR", "Anglosphere", "2609", "02-03-2022-07-09-2022", "CAPI/CAWI/Post/Video_interviewing", "English"],
"GRC": ["18", "Greece", "GRC", "Isolate", "1200", "08-09-2017-16-10-2017", "PAPI", "Greek"],
"GTM": ["20", "Guatemala", "GTM", "Latin", "1203", "03-10-2019-25-02-2020", "CAPI", "Spanish"],
"HKG": ["21", "Hong_Kong_SAR", "HKG", "EastAsia", "2075", "16-07-2018-11-11-2018", "PAPI/CAWI", "Cantonese,English,Putonghua"],
"IDN": ["22", "Indonesia", "IDN", "EastAsia", "3200", "01-06-2018-20-08-2018", "CAPI", "Indonesian"],
"IRN": ["23", "Iran", "IRN", "Indo-Iranian", "1499", "24-03-2020-17-04-2020", "PAPI", "Persian"],
"IRQ": ["24", "Iraq", "IRQ", "Semetic", "1200", "08-06-2018-28-06-2018", "CAPI/PAPI", "Arabic"],
"JOR": ["26", "Jordan", "JOR", "Semetic", "1203", "07-06-2018-14-06-2018", "CAPI", "Arabic"],
"JPN": ["25", "Japan", "JPN", "EastAsia", "1353", "05-09-2019-26-09-2019", "Mail/Post", "Japanese"],
"KAZ": ["27", "Kazakhstan", "KAZ", "Turkic", "1276", "01-10-2018-30-11-2018", "PAPI", "Kazakh,Russian"],
"KAZKZ": ["27", "Kazakhstan", "KAZKZ", "Turkic", "1276", "01-10-2018-30-11-2018", "PAPI", "Kazakh,Russian"],
"KAZRS": ["27", "Kazakhstan", "KAZRS", "Slavic", "1276", "01-10-2018-30-11-2018", "PAPI", "Kazakh,Russian"],
"KEN": ["28", "Kenya", "KEN", "SubSaharanAfrica", "1266", "22-05-2021-22-06-2022", "CAPI", "Swahili"],
"KGZ": ["29", "Kyrgyzstan", "KGZ", "Turkic", "1200", "05-12-2019-28-01-2020", "CAPI", "Kirghiz,Russian"],
"KOR": ["53", "South_Korea", "KOR", "EastAsia", "1245", "24-12-2017-16-01-2018", "CAPI", "Korean"],
"LBN": ["30", "Lebanon", "LBN", "Semetic", "1200", "04-06-2018-18-06-2018", "CAPI", "Arabic"],
"LBY": ["31", "Libya", "LBY", "Semetic", "1196", "12-12-2021-26-01-2022", "CAPI", "Arabic"],
"MAC": ["32", "Macau_SAR", "MAC", "EastAsia", "1023", "03-10-2019-17-12-2019", "CAPI", "Chinese"],
"MAR": ["37", "Morocco", "MAR", "Semetic", "1200", "01-11-2021-19-12-2021", "PAPI", "Arabic"],
"MDV": ["34", "Maldives", "MDV", "Indo-Iranian", "1038", "01-09-2021-01-10-2021", "CAPI", "Dhivehi"],
"MEX": ["35", "Mexico", "MEX", "Latin", "1739", "18-01-2018-02-05-2018", "PAPI", "Spanish"],
"MMR": ["38", "Myanmar", "MMR", "EastAsia", "1200", "17-01-2020-03-03-2020", "CAPI", "Burmese"],
"MNG": ["36", "Mongolia", "MNG", "Turkic", "1638", "04-09-2019-06-02-2021", "CAPI", "Mongolian"],
"MYS": ["33", "Malaysia", "MYS", "EastAsia", "1313", "05-04-2018-21-05-2018", "CAWI/CAPI", "Malay,Chinese"],
"NGA": ["42", "Nigeria", "NGA", "SubSaharanAfrica", "1237", "19-12-2017-26-01-2018", "CAPI", "Hausa,Igbo,Yoruba,English"],
"NIC": ["41", "Nicaragua", "NIC", "Latin", "1200", "30-11-2019-05-01-2020", "CAPI", "Spanish"],
"NIR": ["43", "Northern_Ireland", "NIR", "Anglosphere", "447", "01-03-2022-07-09-2022", "CAPI/CAWI/Post/Video_interviewing", "English"],
"NLD": ["39", "Netherlands", "NLD", "Germanic", "2145", "03-01-2022-25-01-2022", "CAWI", "Dutch"],
"NZL": ["40", "New_Zealand", "NZL", "Anglosphere", "1057", "04-07-2019-21-02-2020", "Mail/Post", "English"],
"PAK": ["44", "Pakistan", "PAK", "Indo-Iranian", "1995", "04-11-2018-11-12-2018", "CAPI", "Urdu"],
"PER": ["45", "Peru", "PER", "Latin", "1400", "17-08-2018-09-09-2018", "PAPI", "Spanish"],
"PHL": ["46", "Philippines", "PHL", "EastAsia", "1200", "03-12-2019-09-12-2019", "PAPI", "Bikol,Cebuano,Filipino,Ikolo,Tausug,Waray,Hiligaynon"],
"PRI": ["47", "Puerto_Rico", "PRI", "Latin", "1127", "16-03-2018-27-10-2018", "PAPI", "Spanish"],
"ROU": ["48", "Romania", "ROU", "Latin", "1257", "30-11-2017-02-04-2018", "CAPI", "Romanian"],
"RUS": ["49", "Russia", "RUS", "Slavic", "1810", "07-11-2017-29-12-2017", "CAPI/PAPI", "Russian"],
"SGP": ["51", "Singapore", "SGP", "EastAsia", "2012", "08-11-2019-15-03-2020", "PAPI", "English,Malay,Chinese"],
"SRB": ["50", "Serbia", "SRB", "Slavic", "1046", "20-05-2017-07-07-2017", "PAPI", "Serbian"],
"SVK": ["52", "Slovakia", "SVK", "Slavic", "1200", "19-01-2022-22-02-2022", "CAPI", "Slovak"],
"THA": ["56", "Thailand", "THA", "EastAsia", "1500", "01-12-2017-26-02-2018", "PAPI", "Thai"],
"TJK": ["55", "Tajikistan", "TJK", "Indo-Iranian", "1200", "08-01-2020-06-02-2020", "CAPI", "Tajik,Russian"],
"TUN": ["57", "Tunisia", "TUN", "Semetic", "1208", "26-04-2019-20-05-2019", "CAPI", "Arabic"],
"TUR": ["58", "Turkey", "TUR", "Turkic", "2415", "31-03-2018-21-05-2018", "PAPI", "Turkish"],
"TWN": ["54", "Taiwan_ROC", "TWN", "EastAsia", "1223", "25-03-2019-16-06-2019", "CAPI", "Chinese"],
"UKR": ["59", "Ukraine", "UKR", "Slavic", "1289", "25-07-2020-14-08-2020", "CAPI", "Ukrainian,Russian"],
"URY": ["61", "Uruguay", "URY", "Latin", "1000", "27-01-2022-22-03-2022", "CAPI", "Spanish"],
"USA": ["60", "United_States", "USA", "Anglosphere", "2596", "28-04-2017-31-05-2017", "CAWI/CATI", "English"],
"USS": ["60", "United_States_South", "USS", "Anglosphere", "2596", "28-04-2017-31-05-2017", "CAWI/CATI", "English"],
"USN": ["60", "United_States_North", "USN", "Anglosphere", "2596", "28-04-2017-31-05-2017", "CAWI/CATI", "English"],
"USD": ["67", "United_States_Democrats", "USD", "Anglosphere", "2596", "28-04-2017-31-05-2017", "CAWI/CATI", "English"],
"USR": ["68", "United_States_Republicans", "USR", "Anglosphere", "2596", "28-04-2017-31-05-2017", "CAWI/CATI", "English"],
"VEN": ["62", "Venezuela", "VEN", "Latin", "1190", "03-05-2021-26-07-2021", "PAPI", "Spanish"],
"VNM": ["63", "Vietnam", "VNM", "EastAsia", "1200", "15-12-2019-21-01-2020", "CAPI", "Vietnamese"],
"ZWE": ["64", "Zimbabwe", "ZWE", "SubSaharanAfrica", "1215", "11-02-2020-23-03-2020", "CAPI", "English,Shona,Ndebele"]
};

#RGB expected format by matplotlib = floats in the range of 0-1
dict_langfam2color={
"Anglosphere": (0.592, 0, 0), #dark red
"EastAsia": (0.4627, 0.8118, 0.8314), #dark turquoise
"Germanic": (0.376, 0.376, 0.376), #grey
"Indo-Iranian": (0.6, 0.298, 0), #dark yellow/orange
"Isolate": (0.4196, 0.3804, 0.5529),  #dark pink
"Latin": (0.6627, 0.6157, 0.0588), #light yellow
"Semetic": (0.2039, 0.3843, 0.0902), #dark green
"Slavic": (0.8314, 0.4627, 0.4980), #dark blue/green
"SubSaharanAfrica": (0.0706, 0.3020, 0.4078), #dark blue
"Turkic": (0.298, 0, 0.6) #dark purple
}


language_families = {
    'AND': 'Latin',
    'ARG': 'Latin',
    'AUS': 'Anglosphere',
    'BGD': 'Indo-Iranian',
    'ARM': 'Isolate',
    'BOL': 'Latin',
    'BRA': 'Latin',
    'MMR': 'EastAsia',
    'CAN': 'Anglosphere',
    'CHL': 'Latin',
    'CHN': 'EastAsia',
    'TWN': 'EastAsia',
    'COL': 'Latin',
    'CYP': 'Isolate',
    'CZE': 'Slavic',
    'ECU': 'Latin',
    'ETH': 'Semetic',
    'DEU': 'Germanic',
    'GRC': 'Isolate',
    'GTM': 'Latin',
    'HKG': 'EastAsia',
    'IDN': 'EastAsia',
    'IRN': 'Indo-Iranian',
    'IRQ': 'Semetic',
    'JPN': 'EastAsia',
    'KAZ': 'Turkic',
    'JOR': 'Semetic',
    'KEN': 'SubSaharanAfrica',
    'KOR': 'EastAsia',
    'KGZ': 'Turkic',
    'LBN': 'Semetic',
    'LBY': 'Semetic',
    'MAC': 'EastAsia',
    'MYS': 'EastAsia',
    'MDV': 'Indo-Iranian',
    'MEX': 'Latin',
    'MNG': 'Turkic',
    'MAR': 'Semetic',
    'NLD': 'Germanic',
    'NZL': 'Anglosphere',
    'NIC': 'Latin',
    'NGA': 'SubSaharanAfrica',
    'PAK': 'Indo-Iranian',
    'PER': 'Latin',
    'PHL': 'EastAsia',
    'PRI': 'Latin',
    'ROU': 'Latin',
    'RUS': 'Slavic',
    'SRB': 'Slavic',
    'SGP': 'EastAsia',
    'SVK': 'Slavic',
    'VNM': 'EastAsia',
    'ZWE': 'SubSaharanAfrica',
    'TJK': 'Indo-Iranian',
    'THA': 'EastAsia',
    'TUN': 'Semetic',
    'TUR': 'Turkic',
    'UKR': 'Slavic',
    'EGY': 'Semetic',
    'GBR': 'Anglosphere',
    'USA': 'Anglosphere',
    'URY': 'Latin',
    'VEN': 'Latin',
    'NIR': 'Anglosphere',
    'CDE': 'Anglosphere',
    'CDF':'Latin',
    'USN':'Anglosphere',
    'USS':'Anglosphere',
    'USD':'Anglosphere',
    'USR':'Anglosphere',
    'KAZKZ':'Turkic',
    'KAZRS':'Slavic',
    'CANCHN':'EastAsia',
    'USASPN':'Latin',
    'GRBPLS':'Slavic',
    'AUSCHN':'EastAsia',
    'NZLCHN':'EastAsia',
    'DEUTRK':'Turkic',
    'NLDENG':'Anglosphere',
}

selected_countries['color'] = selected_countries['iso_a3'].map(language_families).map(dict_langfam2color)
fig, ax = plt.subplots(figsize=(10, 6))
selected_countries.plot(facecolor=selected_countries['color'], ax=ax, edgecolor='black')
unselected_countries.plot(ax=ax, facecolor='lightgrey', edgecolor='black')
ax.set_title('WVS language families')
ax.set_axis_off()
plt.show()


# In[2408]:


language_counts = Counter(language_families.values())
colors = [dict_langfam2color[fam] for fam in language_counts.keys()]
plt.pie(language_counts.values(), labels=language_counts.keys(), autopct='%1.1f%%', colors=colors)
plt.show()


# In[2409]:


dict_questions = {
    "Q1": {"question": "Important in life: Family", "category": "Morality"},
    "Q2": {"question": "Important in life: Friends", "category": "Morality"},
    "Q3": {"question": "Important in life: Leisure time", "category": "Economical"},
    "Q4": {"question": "Important in life: Politics", "category": "Politics"},
    "Q5": {"question": "Important in life: Work", "category": "Economical"},
    "Q6": {"question": "Important in life: Religion", "category": "Religion"},
    "Q7": {"question": "Important child qualities: good manners", "category": "Morality"},
    "Q8": {"question": "Important child qualities: independence", "category": "Morality"},
    "Q9": {"question": "Important child qualities: hard work", "category": "Morality"},
    "Q10": {"question": "Important child qualities: feeling of responsibility", "category": "Morality"},
    "Q11": {"question": "Important child qualities: imagination", "category": "Morality"},
    "Q12": {"question": "Important child qualities: tolerance and respect for other people", "category": "Morality"},
    "Q13": {"question": "Important child qualities: thrift saving money and things", "category": "Economical"},
    "Q14": {"question": "Important child qualities: determination perseverance", "category": "Morality"},
    "Q15": {"question": "Important child qualities: religious faith", "category": "Religion"},
    "Q16": {"question": "Important child qualities: unselfishness", "category": "Morality"},
    "Q17": {"question": "Important child qualities: obedience", "category": "Morality"},
    "Q18": {"question": "Neighbors: Drug addicts", "category": "Morality"},
    "Q19": {"question": "Neighbors: People of a different race", "category": "Morality"},
    "Q20": {"question": "Neighbors: People who have AIDS", "category": "Morality"},
    "Q21": {"question": "Neighbors: Immigrants/foreign workers", "category": "Morality"},
    "Q22": {"question": "Neighbors: Homosexuals", "category": "Morality"},
    "Q23": {"question": "Neighbors: People of a different religion", "category": "Religion"},
    "Q24": {"question": "Neighbors: Heavy drinkers", "category": "Morality"},
    "Q25": {"question": "Neighbors: Unmarried couples living together", "category": "Morality"},
    "Q26": {"question": "Neighbors: People who speak a different language", "category": "Morality"},
    "Q27": {"question": "One of main goals in life has been to make my parents proud", "category": "Morality"},
    "Q28": {"question": "Pre-school child suffers with working mother", "category": "Gender"},
    "Q29": {"question": "Men make better political leaders than women do", "category": "Gender"},
    "Q30": {"question": "University is more important for a boy than for a girl", "category": "Gender"},
    "Q31": {"question": "Men make better business executives than women do", "category": "Gender"},
    "Q32": {"question": "Being a housewife just as fulfilling", "category": "Gender"},
    "Q33": {"question": "Jobs scarce: Men should have more right to a job than women","category": "Gender"},
    "Q33_3": {"question": "Jobs scarce: Men should have more right to a job than women (3-point scale)","category": "Gender"},
    "Q34": {
    "question": "Jobs scarce: Employers should give priority to (nation) people than immigrants",
    "category": "Politics"
    },
    "Q34_3": {
    "question": "Jobs scarce: Employers should give priority to (nation) people than immigrants (3-point scale)",
    "category": "Politics"
    },
    "Q35": {
    "question": "Problem if women have more income than husband",
    "category": "Gender"
    },
    "Q35_3": {
    "question": "Problem if women have more income than husband (3-point scale)",
    "category": "Gender"
    },
    "Q36": {
    "question": "Homosexual couples are as good parents as other couples",
    "category": "Morality"
    },
    "Q37": {
    "question": "Duty towards society to have children",
    "category": "Morality"
    },
    "Q38": {
    "question": "It is children duty to take care of ill parent",
    "category": "Morality"
    },
    "Q39": {
    "question": "People who do not work turn lazy",
    "category": "Economical"
    },
    "Q40": {
    "question": "Work is a duty towards society",
    "category": "Economical"
    },
    "Q41": {
    "question": "Work should always come first even if it means less spare time",
    "category": "Economical"
    },
    "Q42": {
    "question": "Basic kinds of attitudes concerning society",
    "category": "Politics"
    },
    "Q43": {
    "question": "Future changes: Less importance placed on work",
    "category": "Economical"
    },
    "Q44": {
    "question": "Future changes: More emphasis on technology",
    "category": "Economical"
    },
    "Q45": {
    "question": "Future changes: Greater respect for authority",
    "category": "Politics"
    },
    "Q46": {
    "question": "Feeling of happiness",
    "category": "Morality"
    },
    "Q47": {
    "question": "State of health (subjective)",
    "category": "Morality"
    },
    "Q48": {
    "question": "How much freedom of choice and control",
    "category": "Morality"
    },
    "Q49": {
    "question": "Satisfaction with your life",
    "category": "Morality"
    },
    "Q50": {
    "question": "Satisfaction with financial situation of household",
    "category": "Economical"
    },  "Q51": {"question": "Frequency you/family (last 12 month): Gone without enough food to eat", "category": "Economical"},
    "Q52": {"question": "Frequency you/family (last 12 month): Felt unsafe from crime in your own home", "category": "Morality"},
    "Q53": {"question": "Frequency you/family (last 12 month): Gone without needed medicine or treatment that you needed", "category": "Economical"},
    "Q54": {"question": "Frequency you/family (last 12 month): Gone without a cash income", "category": "Economical"},
    "Q55": {"question": "In the last 12 month, how often have you or your family: Gone without a safe shelter over your head", "category": "Economical"},
    "Q56": {"question": "Standard of living comparing with your parents", "category": "Economical"},
    "Q57": {"question": "Most people can be trusted", "category": "Morality"},
    "Q58": {"question": "Trust: Your family", "category": "Morality"},
    "Q59": {"question": "Trust: Your neighborhood", "category": "Morality"},
    "Q60": {"question": "Trust: People you know personally", "category": "Morality"},
    "Q61": {"question": "Trust: People you meet for the first time", "category": "Morality"},
    "Q62": {"question": "Trust: People of another religion", "category": "Morality"},
    "Q63": {"question": "Trust: People of another nationality", "category": "Morality"},
    "Q64": {"question": "Confidence: Churches", "category": "Religion"},
    "Q65": {"question": "Confidence: Armed Forces", "category": "Politics"},
    "Q66": {"question": "Confidence: The Press", "category": "Politics"},
    "Q67": {"question": "Confidence: Television", "category": "Politics"},
    "Q68": {"question": "Confidence: Labor Unions", "category": "Politics"},
    "Q69": {"question": "Confidence: The Police", "category": "Politics"},
    "Q70": {"question": "Confidence: Justice System/Courts", "category": "Politics"},
    "Q71": {"question": "Confidence: The Government", "category": "Politics"},
    "Q72": {"question": "Confidence: The Political Parties", "category": "Politics"},
    "Q73": {"question": "Confidence: Parliament", "category": "Politics"},
    "Q74": {"question": "Confidence: The Civil Services", "category": "Politics"},
    "Q75": {"question": "Confidence: Universities", "category": "Morality"},
    "Q76": {"question": "Confidence: Elections", "category": "Politics"},
    "Q77": {"question": "Confidence: Major Companies", "category": "Economical"},
    "Q78": {"question": "Confidence: Banks", "category": "Economical"},
    "Q79": {"question": "Confidence: The Environmental Protection Movement", "category": "Morality"},
    "Q80": {"question": "Confidence: The Women's Movement", "category": "Gender"},
    "Q81": {"question": "Confidence: Charitable or humanitarian organizations", "category": "Morality"},
    "Q82": {"question": "Confidence: Major regional organization (combined from country-specific)", "category": "Politics"},
    "Q82_NAFTA": {"question": "Confidence: The North American Free Trade Agreement (NAFTA)", "category": "Politics"},
    "Q83": {"question": "Confidence: The United Nations (UN)", "category": "Politics"},
    "Q84": {"question": "Confidence: International Monetary Fund (IMF)", "category": "Religion"},
    "Q85": {"question": "Confidence: International Criminal Court (ICC)", "category": "Politics"},
    "Q86": {"question": "Confidence: North Atlantic Treaty Organization (NATO)", "category": "Politics"},
    "Q87": {"question": "Confidence: The World Bank (WB)", "category": "Economical"},
    "Q88": {"question": "Confidence: The World Health Organization (WHO)", "category": "Morality"},
    "Q89": {"question": "Confidence: The World Trade Organization (WTO)", "category": "Economical"},
    "Q90": {"question": "International organizations: being effective vs being democratic", "category": "Politics"},
    "Q91": {"question": "Countries with the permanent seats on the UN Security Council", "category": "Politics"},
    "Q92": {"question": "Where are the headquarters of the International Monetary Fund (IMF) located?", "category": "Economical"},
    "Q93": {"question": "Which of the following problems does the organization Amnesty International deal with?", "category": "Morality"},
    "Q94": {"question": "Active/Inactive membership: church or religious org", "category": "Religion"},
    "Q94R": {"question": "Active/Inactive membership: church or religious org", "category": "Religion"},
    "Q95": {"question": "Active/Inactive membership: sport or recreational org", "category": "Economical"},
    "Q95R": {"question": "Active/Inactive membership: sport or recreational org", "category": "Economical"},
    "Q96": {"question": "Active/Inactive membership: art, music, educational org", "category": "Morality"},
    "Q96R": {"question": "Active/Inactive membership: art, music, educational org", "category": "Morality"},
    "Q96R": {"question": "Active/Inactive membership: art, music, educational org", "category": "Morality"},
    "Q97": {"question": "Active/Inactive membership: labor union", "category": "Morality"},
    "Q97R": {"question": "Active/Inactive membership: labor union", "category": "Morality"},
    "Q98": {"question": "Active/Inactive membership: political party", "category": "Politics"},
    "Q98R": {"question": "Active/Inactive membership: political party", "category": "Politics"},
    "Q99": {"question": "Active/Inactive membership: environmental organization", "category": "Morality"},
    "Q99R": {"question": "Active/Inactive membership: environmental organization", "category": "Morality"},
    "Q100": {"question": "Active/Inactive membership: professional organization", "category": "Economical"},
    "Q100R": {"question": "Active/Inactive membership: professional organization", "category": "Economical"},
    "Q101": {"question": "Active/Inactive membership: charitable/humanitarian organization", "category": "Morality"},
    "Q101R": {"question": "Active/Inactive membership: charitable/humanitarian organization", "category": "Morality"},
    "Q102": {"question": "Active/Inactive membership: consumer organization", "category": "Morality"},
    "Q102R": {"question": "Active/Inactive membership: consumer organization", "category": "Morality"},
    "Q103": {"question": "Active/Inactive membership: self-help group, mutual aid group", "category": "Morality"},
    "Q103R": {"question": "Active/Inactive membership: self-help group, mutual aid group", "category": "Morality"},
    "Q104": {"question": "Active/Inactive membership: women's group", "category": "Gender"},
    "Q104R": {"question": "Active/Inactive membership: women's group", "category": "Gender"},
    "Q105": {"question": "Active/Inactive membership: other organization", "category": "Morality"},
    "Q105R": {"question": "Active/Inactive membership: other organization", "category": "Morality"},
    "Q106": {"question": "Income equality vs larger income differences", "category": "Economical"},
    "Q107": {"question": "Private vs state ownership of business", "category": "Economical"},
    "Q108": {"question": "Government's vs individual's responsibility", "category": "Politics"},
    "Q109": {"question": "Competition good or harmful", "category": "Economical"},
    "Q110": {"question": "Success: hard work vs luck", "category": "Morality"},
    "Q111": {"question": "Protecting environment vs. Economic growth", "category": "Morality"},
    "Q112": {"question": "Perceptions of corruption in the country", "category": "Politics"},
    "Q113": {"question": "Involved in corruption: State authorities", "category": "Politics"},
    "Q114": {"question": "Involved in corruption: Business executives", "category": "Politics"},
    "Q115": {"question": "Involved in corruption: Local authorities", "category": "Politics"},
    "Q116": {"question": "Involved in corruption: Civil service providers", "category": "Politics"},
    "Q117": {"question": "Involved in corruption: Journalists and media", "category": "Politics"},
    "Q118": {"question": "Frequency ordinary people pay a bribe, give a gift or do a favor to local officials/service providers in order to get services", "category": "Politics"},
    "Q119": {"question": "Degree of agreement: On the whole, women are less corrupt than men", "category": "Gender"},
    "Q120": {"question": "Risk to be held accountable for giving or receiving a bribe", "category": "Politics"},
    "Q121": {"question": "Impact of immigrants on the development of the country", "category": "Politics"},
    "Q122": {"question": "Immigration in your country: Fills useful jobs in the workforce", "category": "Politics"},
    "Q123": {"question": "Immigration in your country: Strengthens cultural diversity", "category": "Politics"},
    "Q124": {"question": "Immigration in your country: Increases the crime rate", "category": "Politics"},
    "Q125": {"question": "Immigration in your country: Gives asylum to political refugees", "category": "Politics"},
    "Q126": {"question": "Immigration in your country: Increases the risks of terrorism", "category": "Politics"},
    "Q127": {"question": "Immigration in your country: Helps poor people establish new lives", "category": "Politics"},
    "Q128": {"question": "Immigration in your country: Increases unemployment", "category": "Politics"},
    "Q129": {"question": "Immigration in your country: Leads to social conflict", "category": "Politics"},
    "Q130": {"question": "Immigration policy preference", "category": "Politics"},
    "Q131": {"question": "Secure in neighborhood", "category": "Security"},
    "Q132": {"question": "Frequency in your neighborhood: Robberies", "category": "Security"},
    "Q133": {"question": "Frequency in your neighborhood: Alcohol consumed in the streets", "category": "Security"},
    "Q134": {"question": "Frequency in your neighborhood: Police or military interfere with people's private life", "category": "Security"},
    "Q135": {"question": "Frequency in your neighborhood: Racist behavior", "category": "Security"},
    "Q136": {"question": "Frequency in your neighborhood: Drug sale in streets", "category": "Security"},
    "Q137": {"question": "Frequency in your neighborhood: Street violence and fights", "category": "Security"},
    "Q138": {"question": "Frequency in your neighborhood: Sexual harassment", "category": "Morality"},
    "Q139": {"question": "Things done for reasons of security: Didn't carry much money", "category": "Security"},
    "Q140": {"question": "Things done for reasons of security: Preferred not to go out at night", "category": "Security"},
    "Q141": {"question": "Things done for reasons of security: Carried a knife, gun or other weapon", "category": "Security"},
    "Q142": {"question": "Worries: Losing my job or not finding a job", "category": "Economical"},
    "Q143": {"question": "Worries: Not being able to give one's children a good education", "category": "Economical"},
    "Q144": {"question": "Respondent was victim of a crime during the past year", "category": "Security"},
    "Q145": {"question": "Respondent's family was victim of a crime during last year", "category": "Security"},
    "Q146": {"question": "Worries: A war involving my country", "category": "Politics"},
    "Q147": {"question": "Worries: A terrorist attack", "category": "Politics"},
    "Q148": {"question": "Worries: A civil war", "category": "Politics"},
    "Q149": {"question": "Freedom and Equality - Which more important", "category": "Morality"},
    "Q150": {"question": "Freedom and security - Which more important", "category": "Morality"},
    "Q151": {"question": "Willingness to fight for country", "category": "Politics"},
    "Q152": {"question": "Aims of country: first choice", "category": "Politics"},
    "Q153": {"question": "Aims of country: second choice", "category": "Politics"},
    "Q154": {"question": "Aims of respondent: first choice", "category": "Politics"},
    "Q155": {"question": "Aims of respondent: second choice", "category": "Politics"},
    "Q156": {"question": "Most important: first choice", "category": "Morality"},
    "Q157": {"question": "Most important: second choice", "category": "Morality"},
    "Q158": {"question": "Science and technology are making our lives healthier, easier, and more comfortable", "category": "Economical"},
    "Q159": {"question": "Because of science and technology, there will be more opportunities for the next generation", "category": "Economical"},
    "Q160": {"question": "We depend too much on science and not enough on faith", "category": "Religion"},
    "Q161": {"question": "One of the bad effects of science is that it breaks down people's ideas of right and wrong", "category": "Religion"},
    "Q162": {"question": "It is not important for me to know about science in my daily life", "category": "Economical"},
"Q163": {"question": "The world is better off, or worse off, because of science and technology", "category": "Economical"},
"Q164": {"question": "Importance of God", "category": "Religion"},
"Q165": {"question": "Believe in: God", "category": "Religion"},
"Q166": {"question": "Believe in: life after death", "category": "Religion"},
"Q167": {"question": "Believe in: hell", "category": "Religion"},
"Q168": {"question": "Believe in: heaven", "category": "Religion"},
"Q169": {"question": "Whenever science and religion conflict, religion is always right", "category": "Religion"},
"Q170": {"question": "The only acceptable religion is my religion", "category": "Religion"},
"Q171": {"question": "How often do you attend religious services", "category": "Religion"},
"Q172": {"question": "How often do you pray", "category": "Religion"},
"Q172R": {"question": "How often do you pray (Constructed)", "category": "Religion"},

"Q173": {"question": "Religious person", "category": "Religion"},
"Q174": {"question": "Meaning of religion: To follow religious norms and ceremonies vs To do good to other people", "category": "Religion"},
"Q175": {"question": "Meaning of religion: To make sense of life after death vs To make sense of life in this world", "category": "Religion"},
"Q176": {"question": "Degree of agreement: Nowadays one often has trouble deciding which moral rules are the right ones to follow", "category": "Morality"},
"Q177": {"question": "Justifiable: Claiming government benefits to which you are not entitled", "category": "Morality"},
"Q178": {"question": "Justifiable: Avoiding a fare on public transport", "category": "Morality"},
"Q179": {"question": "Justifiable: Stealing property", "category": "Morality"},
"Q180": {"question": "Justifiable: Cheating on taxes", "category": "Morality"},
"Q181": {"question": "Justifiable: Someone accepting a bribe in the course of their duties", "category": "Morality"},
"Q182": {"question": "Justifiable: Homosexuality", "category": "Morality"},
"Q183": {"question": "Justifiable: Prostitution", "category": "Morality"},
"Q184": {"question": "Justifiable: Abortion", "category": "Morality"},
"Q185": {"question": "Justifiable: Divorce", "category": "Morality"},
"Q186": {"question": "Justifiable: Sex before marriage", "category": "Morality"},
"Q187": {"question": "Justifiable: Suicide", "category": "Morality"},
"Q188": {"question": "Justifiable: Euthanasia", "category": "Morality"},
"Q189": {"question": "Justifiable: For a man to beat his wife", "category": "Morality"},
"Q190": {"question": "Justifiable: Parents beating children", "category": "Morality"},
"Q191": {"question": "Justifiable: Violence against other people", "category": "Morality"},
"Q192": {"question": "Justifiable: Terrorism as a political, ideological or religious mean", "category": "Politics"},
"Q193": {"question": "Justifiable: Having casual sex", "category": "Morality"},
"Q194": {"question": "Justifiable: Political violence", "category": "Politics"},
"Q195": {"question": "Justifiable: Death penalty", "category": "Politics"},
"Q196": {"question": "Government has the right: Keep people under video surveillance in public areas", "category": "Politics"},
"Q197": {"question": "Government has the right: Monitor all e-mails and any other information exchanged on the Internet", "category": "Politics"},
"Q198": {"question": "Government has the right: Collect information about anyone living in [COUNTRY] without their knowledge", "category": "Politics"},
"Q199": {"question": "Interest in politics", "category": "Politics"},
"Q200": {"question": "How often discusses political matters with friends", "category": "Politics"},
"Q201": {"question": "Information source: Daily newspaper", "category": "Politics"},
"Q202": {"question": "Information source: TV news", "category": "Politics"},
"Q203": {"question": "Information source: Radio news", "category": "Politics"},
"Q204": {"question": "Information source: Mobile phone", "category": "Politics"},
"Q205": {"question": "Information source: Email", "category": "Politics"},
"Q206": {"question": "Information source: Internet", "category": "Politics"},
"Q207": {"question": "Information source: Social media (Facebook, Twitter, etc.)", "category": "Politics"},
"Q208": {"question": "Information source: Talk with friends or colleagues", "category": "Politics"},
"Q209": {"question": "Political action: Signing a petition", "category": "Politics"},
"Q210": {"question": "Political action: Joining in boycotts", "category": "Politics"},
"Q211": {"question": "Political action: Attending lawful/peaceful demonstrations", "category": "Politics"},
"Q212": {"question": "Political action: Joining unofficial strikes", "category": "Politics"},
"Q213": {"question": "Social activism: Donating to a group or campaign", "category": "Politics"},
"Q214": {"question": "Social activism: Contacting a government official", "category": "Politics"},
"Q216": {"question": "Social activism: Encouraging others to vote", "category": "Politics"},
"Q217": {"question": "Political actions online: Searching information about politics and political events", "category": "Politics"},
"Q218": {"question": "Political actions online: Signing an electronic petition", "category": "Politics"},
"Q219": {"question": "Political actions online: Encouraging other people to take any form of political action", "category": "Politics"},
"Q220": {"question": "Political actions online: Organizing political activities, events, protests", "category": "Politics"},
"Q221": {"question": "Vote in elections: local level", "category": "Politics"},
"Q222": {"question": "Vote in elections: national level", "category": "Politics"},
"Q223": {"question": "Which party would you vote for if there were a national election tomorrow", "category": "Politics"},
"Q223_ABREV": {"question": "Party preference Abbreviation", "category": "Politics"},
"Q223_LOCAL": {"question": "Party preference Local name", "category": "Politics"},
"Q224": {"question": "How often in country's elections: Votes are counted fairly", "category": "Politics"},
"Q225": {"question": "How often in country's elections: Opposition candidates are prevented from running", "category": "Politics"},
"Q226": {"question": "How often in country's elections: TV news favors the governing party", "category": "Politics"},
"Q227": {"question": "How often in country's elections: Voters are bribed", "category": "Politics"},
"Q228": {"question": "How often in country's elections: Journalists provide fair coverage of elections", "category": "Politics"},
"Q229": {"question": "How often in country's elections: Election officials are fair", "category": "Politics"},
"Q230": {"question": "How often in country's elections: Rich people buy elections", "category": "Politics"},
"Q231": {"question": "How often in country's elections: Voters are threatened with violence at the polls", "category": "Politics"},
"Q232": {"question": "How often in country's elections: Voters are offered a genuine choice in the elections", "category": "Politics"},
"Q233": {"question": "How often in country's elections: Women have equal opportunities to run the office", "category": "Gender"},
"Q234": {"question": "Some people think that having honest elections makes a lot of difference in their lives; other people think that it doesn't matter much", "category": "Politics"},
"Q235": {"question": "Political system: Having a strong leader who does not have to bother with parliament and elections", "category": "Politics"},
"Q236": {"question": "Political system: Having experts, not government, make decisions according to what they think is best for the country", "category": "Politics"},
"Q237": {"question": "Political system: Having the army rule", "category": "Politics"},
"Q238": {"question": "Political system: Having a democratic political system", "category": "Politics"},
"Q239": {"question": "Political system: Having a system governed by religious law in which there are no political parties or elections", "category": "Religion"},
"Q240": {"question": "Left-right political scale", "category": "Politics"},
"Q241": {"question": "Democracy: Governments tax the rich and subsidize the poor", "category": "Economical"},
"Q242": {"question": "Democracy: Religious authorities interpret the laws", "category": "Politics"},
"Q243": {"question": "Democracy: People choose their leaders in free elections", "category": "Politics"},
"Q244": {"question": "Democracy: People receive state aid for unemployment", "category": "Economical"},
"Q245": {"question": "Democracy: The army takes over when government is incompetent", "category": "Politics"},
"Q246": {"question": "Democracy: Civil rights protect people's liberty against oppression", "category": "Morality"},
"Q247": {"question": "Democracy: The state makes people's incomes equal", "category": "Economical"},
"Q248": {"question": "Democracy: People obey their rulers", "category": "Morality"},
"Q249": {"question": "Democracy: Women have the same rights as men", "category": "Gender"},
"Q250": {"question": "Importance of democracy", "category": "Politics"},
"Q251": {"question": "How democratically is this country being governed today", "category": "Politics"},
"Q252": {"question": "Satisfaction with the political system performance", "category": "Politics"},
"Q253": {"question": "Respect for individual human rights nowadays", "category": "Morality"},
"Q254": {"question": "National pride", "category": "Politics"},
"Q255": {"question": "Feel close to your village, town or city", "category": "Politics"},
"Q256": {"question": "Feel close to your district, region", "category": "Politics"},
"Q257": {"question": "Feel close to your country", "category": "Politics"},
"Q258": {"question": "Feel close to your continent", "category": "Politics"},
"Q259": {"question": "Feel close to the world", "category": "Politics"}    
}
dict_category2color = {
    "Security": (0.10196, 0.15686, 0.12156), 
    "Gender": (0.93725, 0.51372, 0.33333), 
    "Religion": (0.80784, 0.85098, 0.56862), 
    "Politics": (0.75294, 0.90980, 0.97647), 
    "Morality": (0.68627, 0.24705, 0.10196),  
    "Economical": (0.61176, 0.83529, 0.18823),
}


# In[2410]:


categories = [dict_questions[question]['category'] for question in dict_questions.keys()]
category_counts = Counter(categories)
colors = [dict_category2color[category] for category in category_counts.keys()]
plt.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%', colors=colors)
plt.show()


# In[ ]:


#Countries in the Wave 7 survey
print(df['B_COUNTRY_ALPHA'].unique())
#{"HKG": ["21", "Hong_Kong_SAR", "HKG", "EastAsia", "2075", "16-07-2018-11-11-2018", "PAPI/CAWI", "Cantonese,English,Putonghua"], "TUR": ["58", "Turkey", "TUR", "Turkic", "2415", "31-03-2018-21-05-2018", "PAPI", "Turkish"], "THA": ["56", "Thailand", "THA", "EastAsia", "1500", "01-12-2017-26-02-2018", "PAPI", "Thai"], "AND": ["1", "Andorra", "AND", "Latin", "1004", "01-06-2018-22-09-2018", "PAPI", "Catalan,English,Spanish,French"], "TWN": ["54", "Taiwan_ROC", "TWN", "EastAsia", "1223", "25-03-2019-16-06-2019", "CAPI", "Chinese"], "USA": ["60", "United_States", "USA", "Anglosphere", "2596", "28-04-2017-31-05-2017", "CAWI/CATI", "English"], "COL": ["11", "Colombia", "COL", "Latin", "1520", "30-11-2018-22-12-2018", "CAPI", "Spanish"], "DEU": ["17", "Germany", "DEU", "Germanic", "1528", "25-10-2017-31-03-2018", "CAPI", "German"], "MEX": ["35", "Mexico", "MEX", "Latin", "1739", "18-01-2018-02-05-2018", "PAPI", "Spanish"], "SRB": ["50", "Serbia", "SRB", "Slavic", "1046", "20-05-2017-07-07-2017", "PAPI", "Serbian"], "ETH": ["16", "Ethiopia", "ETH", "Semetic", "1230", "06-02-2020-19-03-2020", "CAPI", "Amharic,Oromo,Tigris"], "GBR": ["19", "Great_Britain", "GBR", "Anglosphere", "2609", "02-03-2022-07-09-2022", "CAPI/CAWI/Post/Video_interviewing", "English"], "JPN": ["25", "Japan", "JPN", "EastAsia", "1353", "05-09-2019-26-09-2019", "Mail/Post", "Japanese"], "USN": ["67", "United_States_North", "USN", "Anglosphere", "2596", "28-04-2017-31-05-2017", "CAWI/CATI", "English"], "TJK": ["55", "Tajikistan", "TJK", "Indo-Iranian", "1200", "08-01-2020-06-02-2020", "CAPI", "Tajik,Russian"], "ARG": ["2", "Argentina", "ARG", "Latin", "1003", "04-07-2017-19-07-2017", "PAPI", "Spanish"], "NGA": ["42", "Nigeria", "NGA", "SubSaharanAfrica", "1237", "19-12-2017-26-01-2018", "CAPI", "Hausa,Igbo,Yoruba,English"], "USS": ["68", "United_States_South", "USS", "Anglosphere", "2596", "28-04-2017-31-05-2017", "CAWI/CATI", "English"], "CDE": ["65", "Canada_English", "CDE", "Anglosphere", "0", "11-02-2020-23-03-2020", "CAPI", "English"], "LBN": ["30", "Lebanon", "LBN", "Semetic", "1200", "04-06-2018-18-06-2018", "CAPI", "Arabic"], "CDF": ["66", "Canada_French", "CDF", "Latin", "0", "11-02-2020-23-03-2020", "CAPI", "French"], "ARM": ["4", "Armenia", "ARM", "Isolate", "1223", "07-05-2021-07-06-2021", "CAPI", "Armenian"], "PHL": ["46", "Philippines", "PHL", "EastAsia", "1200", "03-12-2019-09-12-2019", "PAPI", "Bikol,Cebuano,Filipino,Ikolo,Tausug,Waray,Hiligaynon"], "ECU": ["14", "Ecuador", "ECU", "Latin", "1200", "24-01-2018-03-03-2018", "CAPI", "Spanish"], "NIC": ["41", "Nicaragua", "NIC", "Latin", "1200", "30-11-2019-05-01-2020", "CAPI", "Spanish"], "MMR": ["38", "Myanmar", "MMR", "EastAsia", "1200", "17-01-2020-03-03-2020", "CAPI", "Burmese"], "SVK": ["52", "Slovakia", "SVK", "Slavic", "1200", "19-01-2022-22-02-2022", "CAPI", "Slovak"], "NZL": ["40", "New_Zealand", "NZL", "Anglosphere", "1057", "04-07-2019-21-02-2020", "Mail/Post", "English"], "LBY": ["31", "Libya", "LBY", "Semetic", "1196", "12-12-2021-26-01-2022", "CAPI", "Arabic"], "ZWE": ["64", "Zimbabwe", "ZWE", "SubSaharanAfrica", "1215", "11-02-2020-23-03-2020", "CAPI", "English,Shona,Ndebele"], "BRA": ["7", "Brazil", "BRA", "Latin", "1762", "15-05-2018-11-06-2018", "CAPI", "Portuguese"], "NIR": ["43", "Northern_Ireland", "NIR", "Anglosphere", "447", "01-03-2022-07-09-2022", "CAPI/CAWI/Post/Video_interviewing", "English"], "CYP": ["12", "Cyprus", "CYP", "Isolate", "1000", "13-05-2019-04-06-2019", "PAPI", "Greek,Turkish"], "CHL": ["9", "Chile", "CHL", "Latin", "1000", "06-01-2018-05-02-2018", "CAPI", "Spanish"], "EGY": ["15", "Egypt", "EGY", "Semetic", "1200", "22-06-2018-07-07-2018", "CAPI", "Arabic"], "CHN": ["10", "China", "CHN", "EastAsia", "3036", "07-07-2018-12-10-2018", "PAPI", "Chinese"], "SGP": ["51", "Singapore", "SGP", "EastAsia", "2012", "08-11-2019-15-03-2020", "PAPI", "English,Malay,Chinese"], "ROU": ["48", "Romania", "ROU", "Latin", "1257", "30-11-2017-02-04-2018", "CAPI", "Romanian"], "GRC": ["18", "Greece", "GRC", "Isolate", "1200", "08-09-2017-16-10-2017", "PAPI", "Greek"], "PRI": ["47", "Puerto_Rico", "PRI", "Latin", "1127", "16-03-2018-27-10-2018", "PAPI", "Spanish"], "PAK": ["44", "Pakistan", "PAK", "Indo-Iranian", "1995", "04-11-2018-11-12-2018", "CAPI", "Urdu"], "MDV": ["34", "Maldives", "MDV", "Indo-Iranian", "1038", "01-09-2021-01-10-2021", "CAPI", "Dhivehi"], "KAZ": ["27", "Kazakhstan", "KAZ", "Turkic", "1276", "01-10-2018-30-11-2018", "PAPI", "Kazakh,Russian"], "BGD": ["5", "Bangladesh", "BGD", "Indo-Iranian", "1200", "03-12-2018-24-12-2018", "PAPI", "Bengali"], "KEN": ["28", "Kenya", "KEN", "SubSaharanAfrica", "1266", "22-05-2021-22-06-2022", "CAPI", "Swahili"], "RUS": ["49", "Russia", "RUS", "Slavic", "1810", "07-11-2017-29-12-2017", "CAPI/PAPI", "Russian"], "GTM": ["20", "Guatemala", "GTM", "Latin", "1203", "03-10-2019-25-02-2020", "CAPI", "Spanish"], "MYS": ["33", "Malaysia", "MYS", "EastAsia", "1313", "05-04-2018-21-05-2018", "CAWI/CAPI", "Malay,Chinese"], "IDN": ["22", "Indonesia", "IDN", "EastAsia", "3200", "01-06-2018-20-08-2018", "CAPI", "Indonesian"], "JOR": ["26", "Jordan", "JOR", "Semetic", "1203", "07-06-2018-14-06-2018", "CAPI", "Arabic"], "iso3_code": ["#index", "Country/Territory", "iso3_code", "language_family", "Sample", "Fieldwork_period", "Mode", "Languages_fielded"], "CAN": ["8", "Canada", "CAN", "Anglosphere", "4018", "02-10-2020-19-10-2020", "CAWI", "English,French"], "PER": ["45", "Peru", "PER", "Latin", "1400", "17-08-2018-09-09-2018", "PAPI", "Spanish"], "MNG": ["36", "Mongolia", "MNG", "Turkic", "1638", "04-09-2019-06-02-2021", "CAPI", "Mongolian"], "URY": ["61", "Uruguay", "URY", "Latin", "1000", "27-01-2022-22-03-2022", "CAPI", "Spanish"], "KGZ": ["29", "Kyrgyzstan", "KGZ", "Turkic", "1200", "05-12-2019-28-01-2020", "CAPI", "Kirghiz,Russian"], "VNM": ["63", "Vietnam", "VNM", "EastAsia", "1200", "15-12-2019-21-01-2020", "CAPI", "Vietnamese"], "MAC": ["32", "Macau_SAR", "MAC", "EastAsia", "1023", "03-10-2019-17-12-2019", "CAPI", "Chinese"], "CZE": ["13", "Czechia", "CZE", "Slavic", "1200", "11-02-2022-13-05-2022", "CAPI", "Czech"], "NLD": ["39", "Netherlands", "NLD", "Germanic", "2145", "03-01-2022-25-01-2022", "CAWI", "Dutch"], "AUS": ["3", "Australia", "AUS", "Anglosphere", "1813", "06-04-2018-06-08-2018", "Mail/Post", "English"], "BOL": ["6", "Bolivia", "BOL", "Latin", "2067", "18-01-2017-07-03-2017", "CAPI", "Spanish"], "KOR": ["53", "South_Korea", "KOR", "EastAsia", "1245", "24-12-2017-16-01-2018", "CAPI", "Korean"], "MAR": ["37", "Morocco", "MAR", "Semetic", "1200", "01-11-2021-19-12-2021", "PAPI", "Arabic"], "UKR": ["59", "Ukraine", "UKR", "Slavic", "1289", "25-07-2020-14-08-2020", "CAPI", "Ukrainian,Russian"], "IRN": ["23", "Iran", "IRN", "Indo-Iranian", "1499", "24-03-2020-17-04-2020", "PAPI", "Persian"], "VEN": ["62", "Venezuela", "VEN", "Latin", "1190", "03-05-2021-26-07-2021", "PAPI", "Spanish"], "TUN": ["57", "Tunisia", "TUN", "Semetic", "1208", "26-04-2019-20-05-2019", "CAPI", "Arabic"], "IRQ": ["24", "Iraq", "IRQ", "Semetic", "1200", "08-06-2018-28-06-2018", "CAPI/PAPI", "Arabic"]}


# In[ ]:


#how many missing rows?
missing_rows = (df == -5).sum(axis=1)
print("Number of rows with missing/not available values:", (missing_rows > 0).sum())


# In[ ]:


#Calculate the sum of NaN values for each column of our DataFrame
pd.set_option('display.max_columns', None)
df.isnull().sum().to_frame().T

#No NaN values in our dataset


# **COMMON ISSUSES WITH DATA**
# * Inconsistent column names
# * Too much data
# * Missing data
# * Different data types
# * Duplicate data
# * Too much data
# 
# 
# 

# In[2413]:


#change columns to upper case
df.columns=df.columns.str.upper()
print (df.columns)


# In[2414]:


def performAnalyses(df,prefixoutput, d):  
    #-2 means No answer, -4 means Question was not asked in this country, -5 means Missing; Not available

    #Delete from our DataFrame the columns where the proportion of rows with value -4 is greater than 0.8
    #(Where the question has not been asked in more than 80% of countries)

    count_minus_four = df.apply(lambda x: x[x == -4].count(), axis=0)
    prop_minus_four = count_minus_four / len(df)
    columns_to_delete = prop_minus_four[prop_minus_four > 0.8].index
    df = df.drop(columns_to_delete, axis=1)


    #keep ONLY the questions
    df1 = df.loc[:, df.columns.str.startswith(('Q', 'X', 'V'))]
    df1=df1.drop(['Q_MODE', 'VERSION'], axis=1)
    #Filtering values-based questions
    cols_to_drop = np.r_[52:59, 94:97, 146:164, 215:252, 241:254, 270:279, 327:377, [df1.columns.get_loc(col) for col in ['Q112', 'Q118', 'Q120', 'Q234A','Q82_EU']]]
    df1 = df1.drop(df1.columns[cols_to_drop], axis=1)
    df = df.loc[:, ['B_COUNTRY_ALPHA', 'K_DURATION']]
    #DataFrame containing only the countries, K_Duration column and the Questions
    df=pd.concat([df.iloc[:, :2], df1, df.iloc[:, 2:]], axis=1)

    
    
   #Eliminating the demographics and socioeconomic variables (Starting from Q260)
    subset1 = df.iloc[:, 0:1]
    subset2 = df.iloc[:, 2:201] 
    subset = pd.concat([subset1, subset2], axis=1) 
    subset.describe(include='all')
    
    #sets the number of rows and columns to be unlimited, 
    #which means that all rows and columns of a DataFrame will be displayed
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    
    subset[subset.columns[1:]] = subset[subset.columns[1:]].astype(int)
    subset[subset[subset.columns[1:]] < 0] = np.nan
    
    #Scaling all questions so they are all on a scale from 1 to 5
    max_values = subset.max()
    for col in subset.columns[:]:
        if max_values[col] == 10:
            subset[col] = subset[col] / 2
        elif max_values[col] == 4:
            subset[col] = subset[col] * 5/4
        elif max_values[col] == 3:
            subset[col] = subset[col] * 5/3
        elif max_values[col] == 2:
            subset[col].replace({1:5, 2:1}, inplace=True)
        elif max_values[col] == 1:
            subset[col].replace({1:5, 0:1}, inplace=True)
               
    #we tried bfill but backtracked
    #F_subset=subset.bfill(limit=None)
    
    #computing the mean per country
    df_means=subset.groupby(['B_COUNTRY_ALPHA']).mean()
    df_means_cp=subset.groupby(['B_COUNTRY_ALPHA'], as_index=False).mean()
    
    #computing the variance per country
    df_variances=subset.groupby(['B_COUNTRY_ALPHA']).var()
    
    #we are computing the heatmap to see the missing values
    #dfmeans_heatmap=plot_heatmap(df_means)
    #dfmeans_heatmap.savefig(prefix+"_heatmap.png")
    #
    # THIS IS THE RAW HEATMAP
    #
    #plt.figure(figsize = (40,19))
    #heatmap = sns.heatmap(df_means, vmin=0, vmax=5, square=True, cmap="viridis", linewidths=0.1, 
    #         annot=True, annot_kws={"fontsize":3.5},
    #          xticklabels=1, yticklabels=1)
    #heatmap.set(xlabel="QUESTIONS", ylabel="COUNTRIES")
    #plt.savefig(prefixoutput+"_rawheatmap.png")
    #for i, tick_label in enumerate(heatmap.axes.get_yticklabels()):
    #    tick_label.set_color(tick_label_colors[i])
    #plt.show()

    #
    # IMPUTATION using KNN Imputer
    #    
    imputer = KNNImputer(n_neighbors=2)
    df_mean_imp = pd.DataFrame(imputer.fit_transform(df_means),columns=df_means.columns)
    df_mean_imp = df_mean_imp.set_index(df_means.index)
    if d==0:
        #
        # HEATMAP WITH IMPUTED VALUES
        #

        #@MB ALL PLOTS: create dictionary of questions+color like we have for countries
      #X  #@MB ALL PLOTS: make sure that the countries have only 3 letters ex: CAN -> CDE CDF USA = USN USS USD USR KZK KZR
      #X  #@MB ALL PLOTS:Make sure that the new "countries" USD CDF have the colors that correspond
        #@MB ALL PLOTS: use a fixed-width font for the country names
        #@MB low priority: ALL PLOTS: use a fixed-width font and space-pad the questions ex: zero padding: https://exceljet.net/sites/default/files/styles/original_with_watermark/public/images/formulas/pad%20number%20with%20zeros.png
        #@MB ex: "Q1" and "Q200" become: "Q1  " and "Q200"
        #@MB add kmeans clustering (mid priority)
        #@MB add the colors of the countries to the labels             xxxxDONE

        font_path = fm.findfont(fm.FontProperties(family='monospace'))
        font_name = fm.FontProperties(fname=font_path).get_name()

        # Generate the tick label colors
        tick_label_colors = [dict_langfam2color[language_families[code]] for code in df_mean_imp.index]
        Qtick_label_colors = [dict_category2color[dict_questions[question]["category"]] for question in df_mean_imp.columns]

        # Create the heatmap plot
        plt.figure(figsize=(40, 19))
        a = sns.heatmap(df_mean_imp, vmin=0, vmax=5, square=True, cmap="viridis", linewidths=0.1,
                        annot=True, annot_kws={"fontsize": 3.5}, xticklabels=1, yticklabels=1)
        a.set(xlabel="QUESTIONS", ylabel="COUNTRIES")

        # Set the fixed-width font for y-axis and x-axis tick labels
        for i, tick_label in enumerate(a.axes.get_yticklabels()):
            tick_label.set_color(tick_label_colors[i])
            tick_label.set_fontproperties(fm.FontProperties(fname=font_path, size=8)) 
        for i, tick_label in enumerate(a.axes.get_xticklabels()):
            tick_label.set_color(Qtick_label_colors[i])
            tick_label.set_fontproperties(fm.FontProperties(size=8))  

        plt.savefig(prefixoutput+"_heatmap_imp.png")
        plt.show()


        #
        # CLUSTERMAP WITH IMPUTED VALUES
        #
        #@MB add the colors of the countries to the labels              xxxxDONE
        plt.figure(figsize = (40,19))
        g = sns.clustermap(df_mean_imp,figsize=(40, 19), dendrogram_ratio=(.1, .2),vmin=0, vmax=5, cmap="viridis", linewidths=0.1, annot=True, annot_kws={"fontsize":4}, xticklabels=1, yticklabels=1)
        for i, tick_label in enumerate(g.ax_heatmap.get_ymajorticklabels()):
            country_code = tick_label.get_text()
            if country_code in dict_countrycode2info:
                language_family = dict_countrycode2info[country_code][3]
                if language_family in dict_langfam2color:
                    tick_label.set_color(dict_langfam2color[language_family])
            tick_label.set_label("COUNTRIES") # Set y-axis label as "COUNTRIES"
                    
        for i, tick_label in enumerate(g.ax_heatmap.get_xmajorticklabels()):
            question_code = tick_label.get_text()
            if question_code in dict_questions:
                category = dict_questions[question_code]["category"]
                if category in dict_category2color:
                    color = dict_category2color[category]
                    tick_label.set_color(color)
            tick_label.set_label("QUESTIONS")
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), rotation=0)
        plt.savefig(prefixoutput+"_clustermap_imp.png")
        plt.show()


        #
        # CLUSTERMAP WITH IMPUTED VALUES AND LANGUAGE SCALE AND QUESTION CATEGORIES SCALE           xxxDONE
        #

        df_mean_imp["language_family"] = df_mean_imp.index.map(language_families)
        languagefam = df_mean_imp.language_family

        # Create a lookup table that maps unique values in the variable languagefam to color codes from dict_langfam2color
        lut = dict_langfam2color

        # Convert the color codes to RGB values for row colors
        network_pal = sns.color_palette([lut[x] for x in df_mean_imp['language_family'].unique()])
        row_colors = df_mean_imp['language_family'].map(lut)

        # Extract x-axis categories and corresponding colors
        x_axis_categories = [questions_categories[q] for q in df_mean_imp.columns[:-1]]
        x_axis_colors = [dict_category2color[cat] for cat in x_axis_categories]
        x_axis_pal = sns.color_palette(x_axis_colors)

        # Create the clustermap with row colors and x-axis colors
        plt.figure(figsize=(30, 37), dpi=600)
        g = sns.clustermap(df_mean_imp.iloc[:, :-1], row_colors=row_colors, cmap="viridis", yticklabels=1, col_colors=x_axis_pal)

        # Color x-axis labels based on categories
        for tick_label in g.ax_heatmap.axes.get_xticklabels():
            tick_text = tick_label.get_text()
            category = questions_categories[tick_text]
            tick_label.set_color(dict_category2color[category])
            tick_label.set_label("QUESTIONS") # Set x-axis label as "QUESTIONS"
        # Color y-axis labels based on language family
        for tick_label in g.ax_heatmap.axes.get_yticklabels():
            tick_text = tick_label.get_text()
            language_fam_name = df_mean_imp.loc[tick_text, "language_family"]
            tick_label.set_color(lut[language_fam_name])
            tick_label.set_label("COUNTRIES") # Set y-axis label as "COUNTRIES"
        plt.savefig(prefixoutput + "_clustermaplang_imp.png")
        plt.show()


        #
        # PCA WITH IMPUTED VALUES
        #
        # remove last column which is language family
        dict_country2color = {}
        for country, info in dict_countrycode2info.items():
            lang_fam = info[3]
            color = dict_langfam2color[lang_fam]
            dict_country2color[country] = color

        X = df_mean_imp.iloc[:, :-1]
        # Standardize data
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        # Run PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_std)
        fig = plt.figure(figsize=(10, 8))

        # plot scatter with country colors
        colors = [dict_country2color.get(country, (0, 0, 0)) for country in df_mean_imp.index]
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors)

        # plot labels for data points
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        for i, label in enumerate(df_mean_imp.index):
            x = X_pca[i, 0]
            y = X_pca[i, 1]
            plt.annotate(label, (x, y))

        #We add legends for country color codes
        handles = []
        labels = []
        for lang_fam, color in dict_langfam2color.items():
            handle = plt.scatter([], [], c=[color], label=lang_fam)
            handles.append(handle)
            labels.append(lang_fam)
        plt.legend(handles, labels,fontsize='small')

        # add percentage of explained variance for PC1 and PC2                xxXXDONE
        variance_ratios = pca.explained_variance_ratio_
        plt.title(f'PCA ({variance_ratios[0]*100:.2f}% variance explained by PC1, {variance_ratios[1]*100:.2f}% by PC2)')    
        plt.savefig(prefixoutput+"_pca_pc1_pc2_imp.png")
        plt.show()


        #
        #  t-SNE WITH IMPUTED VALUES
        # @MB add a set of dots/ country color 
        #like this: https://s3-eu-west-1.amazonaws.com/ppreviews-plos-725668748/980809/preview.jpg     xxxDONE

        # remove last column which is language family
        X = df_mean_imp.iloc[:, :-1]

        # Create dictionary of country codes to colors
        dict_country2color = {}
        for country, info in dict_countrycode2info.items():
            lang_fam = info[3]
            color = dict_langfam2color[lang_fam]
            dict_country2color[country] = color
        tsne = TSNE(n_components=2, random_state=42)
        embedded_data = tsne.fit_transform(X)
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = [dict_country2color.get(country, (0, 0, 0)) for country in X.index]
        ax.scatter(embedded_data[:, 0], embedded_data[:, 1], c=colors)
        for i, label in enumerate(X.index):
            ax.annotate(label, (embedded_data[i, 0], embedded_data[i, 1]))

        #We add legends for country color codes
        handles = []
        labels = []
        for lang_fam, color in dict_langfam2color.items():
            handle = ax.scatter([], [], c=[color], label=lang_fam)
            handles.append(handle)
            labels.append(lang_fam)
        ax.legend(handles, labels)
        ax.legend(handles, labels,fontsize='small')


        # Add axis labels and title
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        plt.title('t-SNE with Country Colors')
        plt.savefig(prefixoutput+"_tsne_imp.png")
        plt.show()



        #@MB fix the color of countries in nmf
        #@MB using the dictionary of questions, write more representative questions to text file with "prefix"_NMF_#comp_compX_questions.txt, 
        #
        # Non Negative Matrix WITH IMPUTED VALUES
        #


        for n_components in range(2, 5):

            # Initialize NMF model
            nmf_model = NMF(n_components=n_components)

            # remove last column which is language family
            X = df_mean_imp.iloc[:, :-1]

            # Fit NMF model to data
            W = nmf_model.fit_transform(X)
            H = nmf_model.components_


            # Print the top features for each component
            feature_names = df_mean_imp.columns.values
            for i, component in enumerate(H):
                top_features = [feature_names[j] for j in component.argsort()[:-10:-1]]
                print(f"Top features for component {i}: {', '.join(top_features)}")

            print(W)
            print(H)
            W_norm = W / W.sum(axis=1, keepdims=True)
            print(W_norm)

            # Sort the stacked barplot by the values in the first column of W
            sort_idx = np.argsort(W_norm[:, 0])[::-1]
            dftoplot = df_mean_imp.iloc[sort_idx, :]
            W_norm = W_norm[sort_idx, :]
            H = H[:, sort_idx]

            # Create a stacked barplot where each bar sums up to 1
            fig, ax = plt.subplots(figsize=(10, 6))
            ind = np.arange(dftoplot.shape[0])
            width = 0.8

            # Initialize the bottom of each bar to zero
            bottom = np.zeros(dftoplot.shape[0])

            # Plot each component as a stacked bar
            for i in range(n_components):
                ax.bar(ind, W_norm[:, i], width, bottom=bottom, label=f"Component {i}")
                bottom += W_norm[:, i]

            # Set the x-axis labels to the index of the DataFrame and rotate them by 90 degrees
            ax.set_xticks(ind)
            ax.set_xticklabels(dftoplot.index, rotation=90)
            for i, tick_label in enumerate(ax.get_xticklabels()):            
                tick_label.set_color(tick_label_colors[i])    

            # Add a legend and title to the plot
            ax.legend()
            #@MB add the n_components to the title                      XXXXXDONE

            ax.set_title(f"Non-Negative Matrix Factorization (n_components={n_components})")
            #@MB add the country colors to the axis labels                 xxxDONE
            plt.savefig(prefixoutput+"_nmf_c"+str(n_components)+"_imp.png")
            plt.show()
            if n_components == 3:
                # Calculate the coordinates of the vertices of the triangular plot
                x = [0, 1, 0.5]
                y = [0, 0, np.sqrt(3)/2]

                # Create a list of scatter traces for each row in W_norm
                traces = []
                for i in range(W_norm.shape[0]):
                    # Calculate the x and y coordinates of the point for the current row
                    x_coord = W_norm[i, 1] + W_norm[i, 2]/2
                    y_coord = W_norm[i, 2] * np.sqrt(3)/2

                    # Create a scatter trace for the current row
                    traces.append(go.Scatter(x=[x_coord], y=[y_coord], mode='markers+text', 
                                              text=[dftoplot.index[i]], textposition='middle center', 
                                              marker=dict(size=20, color='red')))

                # Create a scatter trace for the vertices of the triangular plot
                traces.append(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=0)))

                # Create a layout for the triangular plot
                layout = go.Layout(title='Non-Negative Matrix Factorization', 
                                   xaxis=dict(range=[-0.1, 1.1], showgrid=False, zeroline=False, showticklabels=False), 
                                   yaxis=dict(range=[-0.1, np.sqrt(3)/2+0.1], showgrid=False, zeroline=False, showticklabels=False), 
                                   shapes=[{'type': 'line', 'x0': x[i], 'y0': y[i], 'x1': x[(i+1)%3], 'y1': y[(i+1)%3]} for i in range(3)], 
                                   margin=dict(l=50, r=50, b=50, t=50))

                # Create a figure with the triangular plot
                fig = go.Figure(data=traces, layout=layout)

                # Show the figure
                #fig.savefig(prefixoutput+"_nmf_c3_triangle_imp.png")
                #@MB let's fix this later, right now I cannot export
                #fig.write_image(prefixoutput+"_nmf_c3_triangle_imp.png", format="jpeg", width=800, height=600)

                fig.show()

        #@MB @GR think about whether eucledian distance is best and how to compute confidence intervals
        #
        # pairwise eucledian distance
        #
        # define a function to compute pairwise Euclidean distance
        def pairwise_euclidean_distance(row_pwed, df_pwed):

            #Compute pairwise Euclidean distance between a given row and all other rows in a pandas DataFrame.
            #Args:
                #row: pandas Series object representing the row for which to compute pairwise distances.
                #df: pandas DataFrame object representing the data set containing all rows.

            #Returns:
                #A NumPy array of Euclidean distances between the given row and all other rows in the DataFrame.
            distances = np.sqrt(np.sum(np.square(df_pwed - row_pwed), axis=1))
            return distances

        # remove last column which is language family
        X = df_mean_imp.iloc[:, :-1]

        # compute pairwise Euclidean distance matrix
        dist_matrix = X.apply(lambda row: pairwise_euclidean_distance(row, X), axis=1)

        # create clustered heatmap using seaborn
        g = sns.clustermap(dist_matrix, cmap='viridis', figsize=(12, 12),xticklabels=1,yticklabels=1)

        #g = sns.clustermap(dist_matrix, cmap='viridis', figsize=(12, 12), xticklabels_rotation=90, yticklabels_rotation=0)

        # set row and column labels
        g.ax_heatmap.set_xlabel("Columns")
        g.ax_heatmap.set_ylabel("Rows")

        #@MB color the country labels
        # display the heatmap
        plt.savefig(prefixoutput+"_nmf_pairwise_dendrogram_imp.png")
        plt.show()


        # create linkage matrix using UPGMA algorithm
        linkage_matrix = linkage(dist_matrix.values, method='average')

        # create dendrogram from linkage matrix
        fig, ax = plt.subplots(figsize=(10, 10))

        dendrogram(linkage_matrix, labels=dist_matrix.index, orientation='left')

        # display the dendrogram
        plt.savefig(prefixoutput+"_nmf_pairwise_upgma_imp.png")
        plt.show()

        return df_mean_imp, df_variances, dist_matrix
    
    else:
        print(df.head())
        


# In[2415]:


#Unfiltered dataframe (with immigrants)
df_means, df_var, df_pwdist=performAnalyses(df,"/home/projects/WVS_project/unfiltered", 0)


# In[2416]:


#Removing all citizens who as well as their parents were not born in the country of origin
df_filter_all=df.loc[(df['Q263'].isin([1, -4])) & (df['Q264'].isin([1, -4])) & (df['Q265'].isin([1, -4]))]


# In[2417]:


#Removing all citizens who do not speak the country's native language at home in high
#immigration countries
#1240: English, 1530:German, 1190:Dutch, 1400:French, 2230: Kazakh, 3630:Russian
countries_q272 = {
    'USA': [1240],
    'AUS': [1240],
    'NZL': [1240],
    'GBR': [1240],
    'DEU': [1530],
    'NLD': [1190],
    'CAN': [1240, 1400],
    'KAZ': [2230,3630],
}
conditions = []
for country, q272_values in countries_q272.items():
    conditions.append((df_filter_all['B_COUNTRY_ALPHA'] == country) & (df_filter_all['Q272'].isin(q272_values)))

condition = pd.concat(conditions, axis=0).any(level=0)

df_filter_all = df_filter_all.loc[condition | (~df_filter_all['B_COUNTRY_ALPHA'].isin(countries_q272.keys()))]


# In[2419]:


#Filtered dataframe based on immigration and language spoken at home
df_means_filtered, df_var_filtered, df_pwdist_filtered=performAnalyses(df_filter_all,"/home/projects/WVS_project/filtered", 0)


# In[2420]:


#Splitting Canada into French Canada and English Canada and splitting Kazakhstan into Russian KZ and Kazakh KZ
df_CANKZ=df_filter_all
df_CANKZ.loc[(df_CANKZ['B_COUNTRY_ALPHA'] == 'CAN') & (df_CANKZ['Q272'] == 1240), 'B_COUNTRY_ALPHA'] = 'CDE'
df_CANKZ.loc[(df_CANKZ['B_COUNTRY_ALPHA'] == 'CAN') & (df_CANKZ['Q272'] == 1400), 'B_COUNTRY_ALPHA'] = 'CDF'
df_CANKZ.loc[(df_CANKZ['B_COUNTRY_ALPHA'] == 'KAZ') & (df_CANKZ['Q272'] == 2230), 'B_COUNTRY_ALPHA'] = 'KAZKZ'
df_CANKZ.loc[(df_CANKZ['B_COUNTRY_ALPHA'] == 'KAZ') & (df_CANKZ['Q272'] == 3630), 'B_COUNTRY_ALPHA'] = 'KAZRS'


# In[2421]:


#Filtered dataframe with a split for countries with more than one official language(CAN and KAZ in our case)
df_means_CANKZ_filter, df_var_CANKZ_filter, df_pwdist_CANKZ_filter=performAnalyses(df_CANKZ,"/home/projects/WVS_project/CANKAZ_filter", 0)


# In[2422]:


#Splitting Canada into French Canada and English Canada and splitting USA into Republicans and Democrats
df_USpolitics=df_filter_all
df_USpolitics.loc[(df_USpolitics['B_COUNTRY_ALPHA'] == 'CAN') & (df_USpolitics['Q272'] == 1240), 'B_COUNTRY_ALPHA'] = 'CDE'
df_USpolitics.loc[(df_USpolitics['B_COUNTRY_ALPHA'] == 'CAN') & (df_USpolitics['Q272'] == 1400), 'B_COUNTRY_ALPHA'] = 'CDF'
df_USpolitics.loc[(df_USpolitics['B_COUNTRY_ALPHA'] == 'USA') & (df_USpolitics['Q223'] ==840001 ), 'B_COUNTRY_ALPHA'] = 'USR'
df_USpolitics.loc[(df_USpolitics['B_COUNTRY_ALPHA'] == 'USA') & (df_USpolitics['Q223'] == 840002), 'B_COUNTRY_ALPHA'] = 'USD'


# In[2423]:


#Filtered dataframe where USA is split based on political parties voters and CAN into French and English
df_means_USPOL_filter, df_var_USPOL_filter, df_pwdist_USPOL_filter=performAnalyses(df_USpolitics,"/home/projects/WVS_project/USAPolitics_filter", 0)


# In[ ]:


#Filtered dataframe keeping only the immigrants
df_onlyIMM=df.loc[(df['Q263'].isin([2,-4])) & (df['Q264'].isin([2, -4])) & (df['Q265'].isin([2, -4]))]
df_means_ALLimm, df_var_ALLimm, df_pwdist_ALLimm=performAnalyses(df_onlyIMM,"/home/projects/WVS_project/ALL_imgrnts")


# In[ ]:


#Filtered dataframe where USA is split into North and South based on latitude and Canada into French and English
df_USLT=df_filter_all
df_USLT.loc[(df_USLT['B_COUNTRY_ALPHA'] == 'CAN') & (df_USLT['Q272'] == 1240), 'B_COUNTRY_ALPHA'] = 'CDE'
df_USLT.loc[(df_USLT['B_COUNTRY_ALPHA'] == 'CAN') & (df_USLT['Q272'] == 1400), 'B_COUNTRY_ALPHA'] = 'CDF'
df_USLT.loc[(df_USLT['B_COUNTRY_ALPHA'] == 'USA') & (df_USLT['O2_LATITUDE'] >= 39 ), 'B_COUNTRY_ALPHA'] = 'USS'
df_USLT.loc[(df_USLT['B_COUNTRY_ALPHA'] == 'USA') & (df_USLT['O2_LATITUDE'] < 39), 'B_COUNTRY_ALPHA'] = 'USN'

df_USLT_filter, df_var_USLT_filter, df_pwdist_USLT_filter=performAnalyses(df_USLT,"/home/projects/WVS_project/USALat_filter", 0)


# In[2441]:


#Filtering dataframe based on second most spoken language in high immigration countries:
df_SecLang=df
df_SecLang.loc[(df_SecLang['B_COUNTRY_ALPHA'] == 'CAN') & (df_SecLang['Q272'] == 1240), 'B_COUNTRY_ALPHA'] = 'CANCHN' #Cantonese
df_SecLang.loc[(df_SecLang['B_COUNTRY_ALPHA'] == 'USA') & (df_SecLang['Q272'] == 1270), 'B_COUNTRY_ALPHA'] = 'USASPN' #Spanish; Castilian
df_SecLang.loc[(df_SecLang['B_COUNTRY_ALPHA'] == 'GBR') & (df_SecLang['Q272'] == 3520), 'B_COUNTRY_ALPHA'] = 'GRBPLS' #Polish
df_SecLang.loc[(df_SecLang['B_COUNTRY_ALPHA'] == 'AUS') & (df_SecLang['Q272'] == 2870), 'B_COUNTRY_ALPHA'] = 'AUSCHN' #Standard Chinese; Mandarin; Putonghua; Guoy
df_SecLang.loc[(df_SecLang['B_COUNTRY_ALPHA'] == 'NZL') & (df_SecLang['Q272'] == 2870), 'B_COUNTRY_ALPHA'] = 'NZLCHN' #Standard Chinese; Mandarin; Putonghua; Guoy
df_SecLang.loc[(df_SecLang['B_COUNTRY_ALPHA'] == 'DEU') & (df_SecLang['Q272'] == 4370), 'B_COUNTRY_ALPHA'] = 'DEUTRK' #Turkish
df_SecLang.loc[(df_SecLang['B_COUNTRY_ALPHA'] == 'NLD') & (df_SecLang['Q272'] == 1240), 'B_COUNTRY_ALPHA'] = 'NLDENG' #English
#Filtered dataframe based on second most spoken language
df_SecLang_filter, df_var_SecLang_filter, df_pwdist_SecLang_filter=performAnalyses(df_SecLang,"/home/projects/WVS_project/SecondLanguageFilter", 0)


# In[2438]:


usa_rows = df_SecLang[df_SecLang['B_COUNTRY_ALPHA'] == 'NLDENG']
print(usa_rows['Q272'].value_counts())


# In[ ]:


#@MB plot as a side by side barplot the eucledian distance between French Canada and English Canada to the USA and divide by 3 age groups
#ex: https://docs.devexpress.com/WindowsForms/2972/controls-and-libraries/chart-control/series-views/2d-series-views/bar-series-views/side-by-side-bar-chart

#@MB plot as a boxplot distance to USA as barplot


# In[ ]:


#Filtered dataframe for age range 16-29
df_young=df_filter_all[df_filter_all['X003R2'] == 1]
df_means_young_filter, df_var_young_filter, df_pwdist_young_filter=performAnalyses(df_young,"/home/projects/WVS_project/16-29_Filter")


# In[ ]:


#Filtered dataframe for age range 30-49
df_m=df_filter_all[df_filter_all['X003R2'] == 2]
df_means_m_filter, df_var_m_filter, df_pwdist_m_filter=performAnalyses(df_m,"/home/projects/WVS_project/30-49_Filter")


# In[ ]:


#Filtered dataframe for age over 50
df_o=df_filter_all[df_filter_all['X003R2'] == 3]
df_means_o_filter, df_var_o_filter, df_pwdist_o_filter=performAnalyses(df_o,"/home/projects/WVS_project/50andOver_Filter")


# In[ ]:


#@MB follow this structure where you do:                    xxxxxxxALL done
# filter and create a new dataframe
# call performAnalysis using the new dataframe
# save means, var and distance to 3 new variables
# the code at the bottom here is good

# perform analyses on all
# do not modify the dataframe df

# filter immigrants, create new dataframe
# perform analyses on no immigrants

# on filterer immigrants, split Canada FR/EN+split KAZ 
# perform analyses on no immigrants+split Canada+split KAZ

# on filterer immigrants+split Canada, split USA north/south
# perform analyses on no immigrants+split Canada+USA north south

# on filterer immigrants+split Canada, split USA Demo/Rep
# perform analyses on no immigrants+split Canada+USA Demo/Rep

# on filterer immigrants, split into young, mid, old
# perform analyses on no immigrants+split young mid old


# In[ ]:


#@MB remove recent and first generation immigrants from high immigration countries 
#(Canada, USA, Germany, UK, Netherlands, Australia, NZ),
#run performAnalyses, save the ouput into 3 distinct variable             xxxxDONE
#@MB immigrant = either parent not born in country does not speak main language at home (except Canada FR/EN)

#@MB take that data and split Canada into French and English, run performAnalyses   xxxDONE

#@MB take that data and split USA on the lattitude (everything below or equal <=39 and strictly above >39)
#USS and USN respectively run performAnalyses      XXXXDONE

#@MB using the data without immigration divide into 3 groups of age, 
#run performAnalyses on each matrix independently    XXXXDONE


# In[ ]:




