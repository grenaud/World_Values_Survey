#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]  # default, always present
import geopandas as gpd
import matplotlib.patches as mpatches

from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from matplotlib.patches import Arrow
import matplotlib.font_manager as fm
import matplotlib.cm as cm


# In[2]:


from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
import matplotlib.colors as colors
from ast import literal_eval
from collections import Counter
import matplotlib.font_manager as fm
import umap
import plotly.graph_objs as go
import ternary
import scipy.stats as stats

from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import dendrogram, to_tree

from scipy.spatial.distance import pdist, squareform
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
#from missingpy import KNNImputer
from sklearn.impute import KNNImputer
import warnings
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")



# In[3]:


from concurrent.futures import ProcessPoolExecutor, as_completed
import ternary
import kaleido
from tabulate import tabulate
import os
import ternary.helpers
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
output_dir = '/home/gabriel/projects/WVS/WVS/'


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


def get_color_from_heatmap(value, min_value, max_value, cmap):
    norm = cm.colors.Normalize(vmin=min_value, vmax=max_value)
    return cmap(norm(value))


df = pd.read_csv("/home/gabriel/projects/WVS/WVS/WVS_Cross-National_Wave_7_csv_v5_0.csv.gz",  compression='gzip', encoding= 'utf8')
df.head()


print (df.columns)
df.shape

df.info()





# In[6]:


dict_countrycode2info={
"AND": ["1", "Andorra", "AND", "Latin", "1004", "01-06-2018-22-09-2018", "PAPI", "Catalan,English,Spanish,French"],
"ARG": ["2", "Argentina", "ARG", "Latin", "1003", "04-07-2017-19-07-2017", "PAPI", "Spanish"],
"ARM": ["4", "Armenia", "ARM", "Other", "1223", "07-05-2021-07-06-2021", "CAPI", "Armenian"],
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
"CYP": ["12", "Cyprus", "CYP", "Other", "1000", "13-05-2019-04-06-2019", "PAPI", "Greek,Turkish"],
"CZE": ["13", "Czechia", "CZE", "Slavic", "1200", "11-02-2022-13-05-2022", "CAPI", "Czech"],
"DEU": ["17", "Germany", "DEU", "Germanic", "1528", "25-10-2017-31-03-2018", "CAPI", "German"],
"ECU": ["14", "Ecuador", "ECU", "Latin", "1200", "24-01-2018-03-03-2018", "CAPI", "Spanish"],
"EGY": ["15", "Egypt", "EGY", "Semetic", "1200", "22-06-2018-07-07-2018", "CAPI", "Arabic"],
"ETH": ["16", "Ethiopia", "ETH", "Semetic", "1230", "06-02-2020-19-03-2020", "CAPI", "Amharic,Oromo,Tigris"],
"GBR": ["19", "Great_Britain", "GBR", "Anglosphere", "2609", "02-03-2022-07-09-2022", "CAPI/CAWI/Post/Video_interviewing", "English"],
"GRC": ["18", "Greece", "GRC", "Other", "1200", "08-09-2017-16-10-2017", "PAPI", "Greek"],
"GTM": ["20", "Guatemala", "GTM", "Latin", "1203", "03-10-2019-25-02-2020", "CAPI", "Spanish"],
"HKG": ["21", "Hong_Kong_SAR", "HKG", "EastAsia", "2075", "16-07-2018-11-11-2018", "PAPI/CAWI", "Cantonese,English,Putonghua"],
"IDN": ["22", "Indonesia", "IDN", "EastAsia", "3200", "01-06-2018-20-08-2018", "CAPI", "Indonesian"],
"IRN": ["23", "Iran", "IRN", "Indo-Iranian", "1499", "24-03-2020-17-04-2020", "PAPI", "Persian"],
"IRQ": ["24", "Iraq", "IRQ", "Semetic", "1200", "08-06-2018-28-06-2018", "CAPI/PAPI", "Arabic"],
"JOR": ["26", "Jordan", "JOR", "Semetic", "1203", "07-06-2018-14-06-2018", "CAPI", "Arabic"],
"JPN": ["25", "Japan", "JPN", "EastAsia", "1353", "05-09-2019-26-09-2019", "Mail/Post", "Japanese"],
"KAZ": ["27", "Kazakhstan", "KAZ", "Turkic", "1276", "01-10-2018-30-11-2018", "PAPI", "Kazakh,Russian"],
"KZK": ["27", "Kazakhstan", "KZK", "Turkic", "1276", "01-10-2018-30-11-2018", "PAPI", "Kazakh,Russian"],
"KZR": ["27", "Kazakhstan", "KZR", "Slavic", "1276", "01-10-2018-30-11-2018", "PAPI", "Kazakh,Russian"],
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
"ZWE": ["64", "Zimbabwe", "ZWE", "SubSaharanAfrica", "1215", "11-02-2020-23-03-2020", "CAPI", "English,Shona,Ndebele"],
"CANCHN": ["65", "Canadian Chinese", "CAN", "EastAsia","0","11-02-2020-23-03-2020", "CAPI", "English"],
"USASPN": ["66", "American Spanish", "USA", "Latin","0","11-02-2020-23-03-2020", "CAPI", "English"],
"GRBPLS": ["67", "Great Britain Polish", "GBR", "Slavic","0","11-02-2020-23-03-2020", "CAPI", "English"],
"AUSCHN": ["68", "Australian Chinese", "AUS", "EastAsia","0","11-02-2020-23-03-2020", "CAPI", "English"],
"NZLCHN": ["69", "New Zealand Chinese", "NZL", "EastAsia","0","11-02-2020-23-03-2020", "CAPI", "English"],
"DEUTRK": ["70", "German Turkish", "DEU", "Turkic","0","11-02-2020-23-03-2020", "CAPI", "English"],
"NLDENG": ["71", "Netherlands English", "NLD", "Anglosphere","0","11-02-2020-23-03-2020", "CAPI", "English"]    
};

# Add column names 
column_names = [
    "ID",
    "Country",
    "ISO3 code",
    "Group",
    "Language code",
    "Date Range",
    "Survey Method",
    "Languages",
]

# Create a list of rows with the key and values from the dictionary
rows = [[key] + value for key, value in dict_countrycode2info.items()]

# Convert the list of rows to a LaTeX table
latex_table = tabulate(rows, headers=column_names, tablefmt="latex_booktabs")

# Save the LaTeX table to a file
with open(output_dir+"countries_info.tex", "w") as f:
    f.write(latex_table)

print("LaTeX table saved to ''"+output_dir+"'countries_info.tex'")


# In[7]:


#RGB expected format by matplotlib = floats in the range of 0-1
dict_langfam2color={
"Anglosphere": (0.592, 0, 0), #dark red
"EastAsia": (0.4627, 0.8118, 0.8314), #dark turquoise
"Germanic": (0.76471, 0.65490, 0.64314), #grey
"Indo-Iranian": (0.6, 0.298, 0), #dark yellow/orange
"Other": (0.4196, 0.3804, 0.5529),  #dark pink
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
    'ARM': 'Other',
    'BOL': 'Latin',
    'BRA': 'Latin',
    'MMR': 'EastAsia',
    'CAN': 'Anglosphere',
    'CHL': 'Latin',
    'CHN': 'EastAsia',
    'TWN': 'EastAsia',
    'COL': 'Latin',
    'CYP': 'Other',
    'CZE': 'Slavic',
    'ECU': 'Latin',
    'ETH': 'Semetic',
    'DEU': 'Germanic',
    'GRC': 'Other',
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
    'KZK':'Turkic',
    'KZR':'Slavic',
    'CANCHN':'EastAsia',
    'USASPN':'Latin',
    'GRBPLS':'Slavic',
    'AUSCHN':'EastAsia',
    'NZLCHN':'EastAsia',
    'DEUTRK':'Turkic',
    'NLDENG':'Anglosphere',
}


world = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/110m_cultural/"
    "ne_110m_admin_0_countries.zip"
)
print(world.shape)
world.head()


# In[8]:


aggr_countries = ["USS","USN","USD","USR","KZK","KZR","CDE","CDF"]
language_families_clean = {k: v for k, v in language_families.items() if k not in aggr_countries}

iso_candidates = ["ADM0_A3", "ISO_A3", "SOV_A3", "WB_A3", "GU_A3"]
iso_col = next((c for c in iso_candidates if c in world.columns), None)
if iso_col is None:
    raise ValueError(f"Could not find an ISO3 column. Available columns: {list(world.columns)}")
print("Using ISO3 column:", iso_col)

world["langfam"] = world[iso_col].map(language_families_clean)
world["color"] = world["langfam"].map(dict_langfam2color)

selected = world[world["langfam"].notna()]
unselected = world[world["langfam"].isna()]

fig, ax = plt.subplots(figsize=(14, 8))
unselected.plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.2)
selected.plot(ax=ax, color=selected["color"], edgecolor="black", linewidth=0.2)

handles = [mpatches.Patch(color=c, label=lf) for lf, c in dict_langfam2color.items()]
ax.legend(handles=handles, title="Language Families", loc="lower left", fontsize=8, title_fontsize=9, frameon=True)

ax.set_title("WVS language families")
ax.set_axis_off()

output_path = os.path.join(output_dir, "language_families_map.png")
fig.savefig(output_path, dpi=1200, bbox_inches="tight")
plt.show()
print("Saved:", output_path)


# In[9]:


# In[43]:


english_proficiency = {
    "NLD": {"country": "Netherlands", "score": 661, "level": "Very High Proficiency"},
    "SGP": {"country": "Singapore", "score": 642, "level": "Very High Proficiency"},
    "AUT": {"country": "Austria", "score": 628, "level": "Very High Proficiency"},
    "NOR": {"country": "Norway", "score": 627, "level": "Very High Proficiency"},
    "DNK": {"country": "Denmark", "score": 625, "level": "Very High Proficiency"},
    "BEL": {"country": "Belgium", "score": 620, "level": "Very High Proficiency"},
    "SWE": {"country": "Sweden", "score": 618, "level": "Very High Proficiency"},
    "FIN": {"country": "Finland", "score": 615, "level": "Very High Proficiency"},
    "PRT": {"country": "Portugal", "score": 614, "level": "Very High Proficiency"},
    "DEU": {"country": "Germany", "score": 613, "level": "Very High Proficiency"},
    "HRV": {"country": "Croatia", "score": 612, "level": "Very High Proficiency"},
    "ZAF": {"country": "South Africa", "score": 609, "level": "Very High Proficiency"},
    "POL": {"country": "Poland", "score": 600, "level": "Very High Proficiency"},
    "GRC": {"country": "Greece", "score": 598, "level": "High Proficiency"},
    "SVK": {"country": "Slovakia", "score": 597, "level": "High Proficiency"},
    "LUX": {"country": "Luxembourg", "score": 596, "level": "High Proficiency"},
    "ROU": {"country": "Romania", "score": 595, "level": "High Proficiency"},
    "HUN": {"country": "Hungary", "score": 590, "level": "High Proficiency"},
    "LTU": {"country": "Lithuania", "score": 589, "level": "High Proficiency"},
    "KEN": {"country": "Kenya", "score": 582, "level": "High Proficiency"},
    "BGR": {"country": "Bulgaria", "score": 581, "level": "High Proficiency"},
    "PHL": {"country": "Philippines", "score": 578, "level": "High Proficiency"},
    "CZE": {"country": "Czech Republic", "score": 575, "level": "High Proficiency"},
    "MYS": {"country": "Malaysia", "score": 574, "level": "High Proficiency"},
    "LVA": {"country": "Latvia", "score": 571, "level": "High Proficiency"},
    "EST": {"country": "Estonia", "score": 570, "level": "High Proficiency"},
    "SRB": {"country": "Serbia", "score": 567, "level": "High Proficiency"},
    "NGA": {"country": "Nigeria", "score": 564, "level": "High Proficiency"},
    "CHE": {"country": "Switzerland", "score": 563, "level": "High Proficiency"},
    "ARG": {"country": "Argentina", "score": 562, "level": "High Proficiency"},
    "HKG": {"country": "Hong Kong", "score": 561, "level": "High Proficiency"},
    "ITA": {"country": "Italy", "score": 548, "level": "Moderate Proficiency"},
    "ESP": {"country": "Spain", "score": 545, "level": "Moderate Proficiency"},
    "FRA": {"country": "France", "score": 541, "level": "Moderate Proficiency"},
    "UKR": {"country": "Ukraine", "score": 539, "level": "Moderate Proficiency"},
    "KOR": {"country": "South Korea", "score": 537, "level": "Moderate Proficiency"},
    "CRI": {"country": "Costa Rica", "score": 536, "level": "Moderate Proficiency"},
    "CUB": {"country": "Cuba", "score": 535, "level": "Moderate Proficiency"},
    "BLR": {"country": "Belarus", "score": 533, "level": "Moderate Proficiency"},
    "RUS": {"country": "Russia", "score": 530, "level": "Moderate Proficiency"},
    "GHA": {"country": "Ghana", "score": 529, "level": "Moderate Proficiency"},
    "MDA": {"country": "Moldova", "score": 528, "level": "Moderate Proficiency"},
    "PRY": {"country": "Paraguay", "score": 526, "level": "Moderate Proficiency"},
    "BOL": {"country": "Bolivia", "score": 525, "level": "Moderate Proficiency"},
    "CHL": {"country": "Chile", "score": 524, "level": "Moderate Proficiency"},
    "GEO": {"country": "Georgia", "score": 524, "level": "Moderate Proficiency"},
    "ALB": {"country": "Albania", "score": 523, "level": "Moderate Proficiency"},
    "HND": {"country": "Honduras", "score": 522, "level": "Moderate Proficiency"},
    "URY": {"country": "Uruguay", "score": 521, "level": "Moderate Proficiency"},
    "SLV": {"country": "El Salvador", "score": 519, "level": "Moderate Proficiency"},
    "PER": {"country": "Peru", "score": 517, "level": "Moderate Proficiency"},
    "IND": {"country": "India", "score": 516, "level": "Moderate Proficiency"},
    "DOM": {"country": "Dominican Republic", "score": 514, "level": "Moderate Proficiency"},
    "LBN": {"country": "Lebanon", "score": 513, "level": "Moderate Proficiency"},
    "UGA": {"country": "Uganda", "score": 512, "level": "Moderate Proficiency"},
    "TUN": {"country": "Tunisia", "score": 511, "level": "Moderate Proficiency"},
    "ARM": {"country": "Armenia", "score": 506, "level": "Moderate Proficiency"},
    "BRA": {"country": "Brazil", "score": 505, "level": "Moderate Proficiency"},
    "GTM": {"country": "Guatemala", "score": 505, "level": "Moderate Proficiency"},
    "VNM": {"country": "Vietnam", "score": 502, "level": "Moderate Proficiency"},
    "NIC": {"country": "Nicaragua", "score": 499, "level": "Low Proficiency"},
    "CHN": {"country": "China", "score": 498, "level": "Low Proficiency"},
    "TZA": {"country": "Tanzania", "score": 496, "level": "Low Proficiency"},
    "TUR": {"country": "Turkey", "score": 495, "level": "Low Proficiency"},
    "NPL": {"country": "Nepal", "score": 494, "level": "Low Proficiency"},
    "BGD": {"country": "Bangladesh", "score": 493, "level": "Low Proficiency"},
    "VEN": {"country": "Venezuela", "score": 492, "level": "Low Proficiency"},
    "ETH": {"country": "Ethiopia", "score": 490, "level": "Low Proficiency"},
    "IRN": {"country": "Iran", "score": 489, "level": "Low Proficiency"},
    "PAK": {"country": "Pakistan", "score": 488, "level": "Low Proficiency"},
    "LKA": {"country": "Sri Lanka", "score": 487, "level": "Low Proficiency"},
    "MNG": {"country": "Mongolia", "score": 485, "level": "Low Proficiency"},
    "QAT": {"country": "Qatar", "score": 484, "level": "Low Proficiency"},
    "ISR": {"country": "Israel", "score": 483, "level": "Low Proficiency"},
    "PAN": {"country": "Panama", "score": 482, "level": "Low Proficiency"},
    "MAR": {"country": "Morocco", "score": 478, "level": "Low Proficiency"},
    "COL": {"country": "Colombia", "score": 477, "level": "Low Proficiency"},
    "DZA": {"country": "Algeria", "score": 476, "level": "Low Proficiency"},
    "ARE": {"country": "United Arab Emirates", "score": 476, "level": "Low Proficiency"},
    "JPN": {"country": "Japan", "score": 475, "level": "Low Proficiency"},
    "IDN": {"country": "Indonesia", "score": 469, "level": "Low Proficiency"},
    "ECU": {"country": "Ecuador", "score": 466, "level": "Low Proficiency"},
    "SYR": {"country": "Syria", "score": 461, "level": "Low Proficiency"},
    "KWT": {"country": "Kuwait", "score": 459, "level": "Low Proficiency"},
    "EGY": {"country": "Egypt", "score": 454, "level": "Low Proficiency"},
    "MOZ": {"country": "Mozambique", "score": 453, "level": "Low Proficiency"},
    "AFG": {"country": "Afghanistan", "score": 450, "level": "Low Proficiency"},
    "MEX": {"country": "Mexico", "score": 447, "level": "Very Low Proficiency"},
    "UZB": {"country": "Uzbekistan", "score": 446, "level": "Very Low Proficiency"},
    "JOR": {"country": "Jordan", "score": 443, "level": "Very Low Proficiency"},
    "KGZ": {"country": "Kyrgyzstan", "score": 442, "level": "Very Low Proficiency"},
    "AZE": {"country": "Azerbaijan", "score": 440, "level": "Very Low Proficiency"},
    "MMR": {"country": "Myanmar", "score": 437, "level": "Very Low Proficiency"},
    "KHM": {"country": "Cambodia", "score": 434, "level": "Very Low Proficiency"},
    "SDN": {"country": "Sudan", "score": 426, "level": "Very Low Proficiency"},
    "CMR": {"country": "Cameroon", "score": 425, "level": "Very Low Proficiency"},
    "THA": {"country": "Thailand", "score": 423, "level": "Very Low Proficiency"},
    "HTI": {"country": "Haiti", "score": 421, "level": "Very Low Proficiency"},
    "KAZ": {"country": "Kazakhstan", "score": 420, "level": "Very Low Proficiency"},
    "SOM": {"country": "Somalia", "score": 414, "level": "Very Low Proficiency"},
    "OMN": {"country": "Oman", "score": 412, "level": "Very Low Proficiency"},
    "SAU": {"country": "Saudi Arabia", "score": 406, "level": "Very Low Proficiency"},
    "IRQ": {"country": "Iraq", "score": 404, "level": "Very Low Proficiency"},
    "CIV": {"country": "Ivory Coast", "score": 403, "level": "Very Low Proficiency"},
    "AGO": {"country": "Angola", "score": 402, "level": "Very Low Proficiency"},
    "TJK": {"country": "Tajikistan", "score": 397, "level": "Very Low Proficiency"},
    "RWA": {"country": "Rwanda", "score": 392, "level": "Very Low Proficiency"},
    "LBY": {"country": "Libya", "score": 390, "level": "Very Low Proficiency"},
    "YEM": {"country": "Yemen", "score": 370, "level": "Very Low Proficiency"},
    "COD": {"country": "Democratic Republic of the Congo", "score": 367, "level": "Very Low Proficiency"},
    "LAO": {"country": "Laos", "score": 364, "level": "Very Low Proficiency"}
}


# In[44]:


language_counts = Counter(language_families.values())
colors = [dict_langfam2color[fam] for fam in language_counts.keys()]
plt.pie(language_counts.values(), labels=language_counts.keys(), autopct='%1.1f%%', colors=colors)
plt.show()
output_filename = 'language_families_pie_chart.png'  
output_path = output_dir + output_filename
plt.savefig(output_path, dpi=300)


# In[10]:


cols_to_remove = ['Q50', 'Q51', 'Q52', 'Q53', 'Q54', 'Q55', 'Q56','Q82_EU', 'Q91', 'Q92', 'Q93','Q112', 'Q118','Q120','Q131', 'Q132', 'Q133', 'Q134', 'Q135', 'Q136', 'Q137', 'Q138', 'Q139', 'Q140', 'Q141', 'Q142', 'Q143', 'Q144', 'Q145', 'Q146', 'Q147', 'Q148', 'Q199', 'Q200', 'Q201', 'Q202', 'Q203', 'Q204', 'Q205', 'Q206', 'Q207', 'Q208', 'Q209', 'Q210', 'Q211', 'Q212', 'Q213', 'Q214', 'Q215', 'Q216', 'Q217', 'Q218', 'Q219', 'Q220', 'Q221', 'Q222', 'Q223', 'Q223_ABREV', 'Q223_LOCAL', 'Q224', 'Q225', 'Q226', 'Q227', 'Q228', 'Q229', 'Q230', 'Q231', 'Q232', 'Q233', 'Q223_LOCAL', 'Q224', 'Q234A', 'Q225', 'Q226', 'Q227', 'Q228', 'Q229', 'Q230', 'Q231', 'Q232', 'Q233', 'Q234', 'Q234A', 'Q251', 'Q252', 'Q253', 'Q254', 'Q255', 'Q256', 'Q257', 'Q258', 'Q259', 'Q291UN3', 'Q291UN5', 'VOICE', 'V2X_POLYARCHY', 'V2X_LIBDEM', 'V2X_PARTIPDEM', 'V2X_DELIBDEM', 'V2X_EGALDEM', 'V2X_FREEXP_ALTINF', 'V2X_FRASSOC_THICK', 'V2XEL_FREFAIR', 'V2XCL_ROL', 'V2X_CSPART', 'V2XEG_EQDR', 'V2EXCRPTPS', 'V2EXTHFTPS', 'V2JUACCNT', 'V2CLTRNSLW', 'V2CLACJUST', 'V2CLSOCGRP', 'V2CLACFREE', 'V2CLRELIG', 'V2CSRLGREP', 'V2MECENEFM', 'V2MECENEFI', 'V2MEBIAS', 'V2PEPWRSES', 'V2PEPWRGEN', 'V2PEEDUEQ', 'V2PEHEALTH', 'V2PEAPSECON', 'V2PEASJSOECON', 'V2CLGENCL', 'V2PEASJGEN', 'V2PEASBGEN', 'V2CAFRES', 'V2CAFEXCH', 'V2X_CORR', 'V2X_GENDER', 'V2X_GENCL', 'V2X_GENPP', 'V2X_RULE', 'V2XCL_ACJST', 'V2PSBARS', 'V2PSORGS', 'V2PSPRBRCH', 'V2PSPRLNKS', 'V2PSPLATS', 'V2XNP_CLIENT', 'V2XPS_PARTY'];

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
    },  
    "Q51": {"question": "Frequency you/family (last 12 month): Gone without enough food to eat", "category": "Economical"},
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
    "Security": (0.30980, 0.50588, 0.74118),
    "Gender": (0.82353, 0.49020, 0.78039),
    "Religion": (0.45098, 0.57255, 0.41176),

    "Politics": ((0.92549, 0.41961, 0.33725)), 
    "Morality": (1.00000, 0.75686, 0.32941),  
    "Economical": (0.27843, 0.70196, 0.61176),

}

latex_table = "\\begin{tabular}{|c|l|l|c|}\n\\hline\n"
latex_table += "\\textbf{Key} & \\textbf{Question} & \\textbf{Category} & \\textbf{Included} \\\\\n\\hline\n"

for key, values in dict_questions.items():
    included = "No" if key in cols_to_remove else "Yes"
    latex_table += f"{key} & {values['question']} & {values['category']} & {included} \\\\\n\\hline\n"

latex_table += "\\end{tabular}"


with open(output_dir+"questions_latex_table.tex", "w") as f:
    f.write(latex_table)


# In[11]:


categories = [dict_questions[question]['category'] for question in dict_questions.keys()]
category_counts = Counter(categories)
colors = [dict_category2color[category] for category in category_counts.keys()]
plt.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%', colors=colors)
output_filename = 'questions_categories_pie_chart.png'  
output_path = output_dir + output_filename
plt.savefig(output_path, dpi=1200)
plt.show()


# In[12]:


df.rename(columns={'B_COUNTRY_ALPHA': 'COUNTRIES'}, inplace=True)


# In[48]:


print(df['COUNTRIES'].unique())
#{"HKG": ["21", "Hong_Kong_SAR", "HKG", "EastAsia", "2075", "16-07-2018-11-11-2018", "PAPI/CAWI", "Cantonese,English,Putonghua"], "TUR": ["58", "Turkey", "TUR", "Turkic", "2415", "31-03-2018-21-05-2018", "PAPI", "Turkish"], "THA": ["56", "Thailand", "THA", "EastAsia", "1500", "01-12-2017-26-02-2018", "PAPI", "Thai"], "AND": ["1", "Andorra", "AND", "Latin", "1004", "01-06-2018-22-09-2018", "PAPI", "Catalan,English,Spanish,French"], "TWN": ["54", "Taiwan_ROC", "TWN", "EastAsia", "1223", "25-03-2019-16-06-2019", "CAPI", "Chinese"], "USA": ["60", "United_States", "USA", "Anglosphere", "2596", "28-04-2017-31-05-2017", "CAWI/CATI", "English"], "COL": ["11", "Colombia", "COL", "Latin", "1520", "30-11-2018-22-12-2018", "CAPI", "Spanish"], "DEU": ["17", "Germany", "DEU", "Germanic", "1528", "25-10-2017-31-03-2018", "CAPI", "German"], "MEX": ["35", "Mexico", "MEX", "Latin", "1739", "18-01-2018-02-05-2018", "PAPI", "Spanish"], "SRB": ["50", "Serbia", "SRB", "Slavic", "1046", "20-05-2017-07-07-2017", "PAPI", "Serbian"], "ETH": ["16", "Ethiopia", "ETH", "Semetic", "1230", "06-02-2020-19-03-2020", "CAPI", "Amharic,Oromo,Tigris"], "GBR": ["19", "Great_Britain", "GBR", "Anglosphere", "2609", "02-03-2022-07-09-2022", "CAPI/CAWI/Post/Video_interviewing", "English"], "JPN": ["25", "Japan", "JPN", "EastAsia", "1353", "05-09-2019-26-09-2019", "Mail/Post", "Japanese"], "USN": ["67", "United_States_North", "USN", "Anglosphere", "2596", "28-04-2017-31-05-2017", "CAWI/CATI", "English"], "TJK": ["55", "Tajikistan", "TJK", "Indo-Iranian", "1200", "08-01-2020-06-02-2020", "CAPI", "Tajik,Russian"], "ARG": ["2", "Argentina", "ARG", "Latin", "1003", "04-07-2017-19-07-2017", "PAPI", "Spanish"], "NGA": ["42", "Nigeria", "NGA", "SubSaharanAfrica", "1237", "19-12-2017-26-01-2018", "CAPI", "Hausa,Igbo,Yoruba,English"], "USS": ["68", "United_States_South", "USS", "Anglosphere", "2596", "28-04-2017-31-05-2017", "CAWI/CATI", "English"], "CDE": ["65", "Canada_English", "CDE", "Anglosphere", "0", "11-02-2020-23-03-2020", "CAPI", "English"], "LBN": ["30", "Lebanon", "LBN", "Semetic", "1200", "04-06-2018-18-06-2018", "CAPI", "Arabic"], "CDF": ["66", "Canada_French", "CDF", "Latin", "0", "11-02-2020-23-03-2020", "CAPI", "French"], "ARM": ["4", "Armenia", "ARM", "Isolate", "1223", "07-05-2021-07-06-2021", "CAPI", "Armenian"], "PHL": ["46", "Philippines", "PHL", "EastAsia", "1200", "03-12-2019-09-12-2019", "PAPI", "Bikol,Cebuano,Filipino,Ikolo,Tausug,Waray,Hiligaynon"], "ECU": ["14", "Ecuador", "ECU", "Latin", "1200", "24-01-2018-03-03-2018", "CAPI", "Spanish"], "NIC": ["41", "Nicaragua", "NIC", "Latin", "1200", "30-11-2019-05-01-2020", "CAPI", "Spanish"], "MMR": ["38", "Myanmar", "MMR", "EastAsia", "1200", "17-01-2020-03-03-2020", "CAPI", "Burmese"], "SVK": ["52", "Slovakia", "SVK", "Slavic", "1200", "19-01-2022-22-02-2022", "CAPI", "Slovak"], "NZL": ["40", "New_Zealand", "NZL", "Anglosphere", "1057", "04-07-2019-21-02-2020", "Mail/Post", "English"], "LBY": ["31", "Libya", "LBY", "Semetic", "1196", "12-12-2021-26-01-2022", "CAPI", "Arabic"], "ZWE": ["64", "Zimbabwe", "ZWE", "SubSaharanAfrica", "1215", "11-02-2020-23-03-2020", "CAPI", "English,Shona,Ndebele"], "BRA": ["7", "Brazil", "BRA", "Latin", "1762", "15-05-2018-11-06-2018", "CAPI", "Portuguese"], "NIR": ["43", "Northern_Ireland", "NIR", "Anglosphere", "447", "01-03-2022-07-09-2022", "CAPI/CAWI/Post/Video_interviewing", "English"], "CYP": ["12", "Cyprus", "CYP", "Isolate", "1000", "13-05-2019-04-06-2019", "PAPI", "Greek,Turkish"], "CHL": ["9", "Chile", "CHL", "Latin", "1000", "06-01-2018-05-02-2018", "CAPI", "Spanish"], "EGY": ["15", "Egypt", "EGY", "Semetic", "1200", "22-06-2018-07-07-2018", "CAPI", "Arabic"], "CHN": ["10", "China", "CHN", "EastAsia", "3036", "07-07-2018-12-10-2018", "PAPI", "Chinese"], "SGP": ["51", "Singapore", "SGP", "EastAsia", "2012", "08-11-2019-15-03-2020", "PAPI", "English,Malay,Chinese"], "ROU": ["48", "Romania", "ROU", "Latin", "1257", "30-11-2017-02-04-2018", "CAPI", "Romanian"], "GRC": ["18", "Greece", "GRC", "Isolate", "1200", "08-09-2017-16-10-2017", "PAPI", "Greek"], "PRI": ["47", "Puerto_Rico", "PRI", "Latin", "1127", "16-03-2018-27-10-2018", "PAPI", "Spanish"], "PAK": ["44", "Pakistan", "PAK", "Indo-Iranian", "1995", "04-11-2018-11-12-2018", "CAPI", "Urdu"], "MDV": ["34", "Maldives", "MDV", "Indo-Iranian", "1038", "01-09-2021-01-10-2021", "CAPI", "Dhivehi"], "KAZ": ["27", "Kazakhstan", "KAZ", "Turkic", "1276", "01-10-2018-30-11-2018", "PAPI", "Kazakh,Russian"], "BGD": ["5", "Bangladesh", "BGD", "Indo-Iranian", "1200", "03-12-2018-24-12-2018", "PAPI", "Bengali"], "KEN": ["28", "Kenya", "KEN", "SubSaharanAfrica", "1266", "22-05-2021-22-06-2022", "CAPI", "Swahili"], "RUS": ["49", "Russia", "RUS", "Slavic", "1810", "07-11-2017-29-12-2017", "CAPI/PAPI", "Russian"], "GTM": ["20", "Guatemala", "GTM", "Latin", "1203", "03-10-2019-25-02-2020", "CAPI", "Spanish"], "MYS": ["33", "Malaysia", "MYS", "EastAsia", "1313", "05-04-2018-21-05-2018", "CAWI/CAPI", "Malay,Chinese"], "IDN": ["22", "Indonesia", "IDN", "EastAsia", "3200", "01-06-2018-20-08-2018", "CAPI", "Indonesian"], "JOR": ["26", "Jordan", "JOR", "Semetic", "1203", "07-06-2018-14-06-2018", "CAPI", "Arabic"], "iso3_code": ["#index", "Country/Territory", "iso3_code", "language_family", "Sample", "Fieldwork_period", "Mode", "Languages_fielded"], "CAN": ["8", "Canada", "CAN", "Anglosphere", "4018", "02-10-2020-19-10-2020", "CAWI", "English,French"], "PER": ["45", "Peru", "PER", "Latin", "1400", "17-08-2018-09-09-2018", "PAPI", "Spanish"], "MNG": ["36", "Mongolia", "MNG", "Turkic", "1638", "04-09-2019-06-02-2021", "CAPI", "Mongolian"], "URY": ["61", "Uruguay", "URY", "Latin", "1000", "27-01-2022-22-03-2022", "CAPI", "Spanish"], "KGZ": ["29", "Kyrgyzstan", "KGZ", "Turkic", "1200", "05-12-2019-28-01-2020", "CAPI", "Kirghiz,Russian"], "VNM": ["63", "Vietnam", "VNM", "EastAsia", "1200", "15-12-2019-21-01-2020", "CAPI", "Vietnamese"], "MAC": ["32", "Macau_SAR", "MAC", "EastAsia", "1023", "03-10-2019-17-12-2019", "CAPI", "Chinese"], "CZE": ["13", "Czechia", "CZE", "Slavic", "1200", "11-02-2022-13-05-2022", "CAPI", "Czech"], "NLD": ["39", "Netherlands", "NLD", "Germanic", "2145", "03-01-2022-25-01-2022", "CAWI", "Dutch"], "AUS": ["3", "Australia", "AUS", "Anglosphere", "1813", "06-04-2018-06-08-2018", "Mail/Post", "English"], "BOL": ["6", "Bolivia", "BOL", "Latin", "2067", "18-01-2017-07-03-2017", "CAPI", "Spanish"], "KOR": ["53", "South_Korea", "KOR", "EastAsia", "1245", "24-12-2017-16-01-2018", "CAPI", "Korean"], "MAR": ["37", "Morocco", "MAR", "Semetic", "1200", "01-11-2021-19-12-2021", "PAPI", "Arabic"], "UKR": ["59", "Ukraine", "UKR", "Slavic", "1289", "25-07-2020-14-08-2020", "CAPI", "Ukrainian,Russian"], "IRN": ["23", "Iran", "IRN", "Indo-Iranian", "1499", "24-03-2020-17-04-2020", "PAPI", "Persian"], "VEN": ["62", "Venezuela", "VEN", "Latin", "1190", "03-05-2021-26-07-2021", "PAPI", "Spanish"], "TUN": ["57", "Tunisia", "TUN", "Semetic", "1208", "26-04-2019-20-05-2019", "CAPI", "Arabic"], "IRQ": ["24", "Iraq", "IRQ", "Semetic", "1200", "08-06-2018-28-06-2018", "CAPI/PAPI", "Arabic"]}


# In[49]:


#how many missing rows?
missing_rows = (df == -5).sum(axis=1)
print("Number of rows with missing/not available values:", (missing_rows > 0).sum())



# In[13]:


#Calculate the sum of NaN values for each column of our DataFrame
pd.set_option('display.max_columns', None)
df.isnull().sum().to_frame().T
#No NaN values in our dataset



# In[14]:


#change columns to upper case
df.columns=df.columns.str.upper()
print (df.columns)


# In[15]:


df_og=df


# In[16]:


def plot_eng_correlation(english_proficiency_df, cntry, prefixoutput):

    try:

        # Scatter plot
        #sns.scatterplot(x='score', y='distance', data=english_proficiency_df, ax=ax1)

        # Create a scatter plot
        fig, ax1 = plt.subplots(figsize=(12, 6))


        # Convert 'score' column to numeric


        # Scatter plot
        sns.scatterplot(x='score', y='distance', data=english_proficiency_df, ax=ax1)

        # Regression line with confidence interval
        sns.regplot(x='score', y='distance', data=english_proficiency_df, ax=ax1, color='r')

        # Calculate correlation factor
        corr_factor, p_value = stats.pearsonr(english_proficiency_df['score'], english_proficiency_df['distance'])

        # Create a legend with correlation factor
        legend_text = f'Correlation: {corr_factor:.2f}\np-value: {p_value:.3f}'
        ax1.legend([legend_text], loc='upper right')

        # Add labels and title
        ax1.set_xlabel('English Proficiency Score')
        ax1.set_ylabel(f'Distance from {cntry}')
        ax1.set_title(f'English Proficiency Score vs cultural distance from {dict_countrycode2info[cntry][1]}')
        #ax1.set_xlim([400, 800]) 
        #ax1.set_ylim([0, 20]) 
        # Save and show the figure
        plt.savefig(f"{prefixoutput}_english_proficiency_vs_distance_{cntry}.png", dpi=1200)
        plt.show()
        return corr_factor, p_value
    except Exception as e:
        print(f"An error occurred: {e}")


# In[17]:


def create_bar_plot(
    cntry,
    dist_matrix,
    dict_countrycode2info,
    dict_langfam2color,
    ci_lower,
    ci_upper,
    prefixoutput,
):
    """
    Bar plot of distances from `cntry` to all other countries with CI error bars.

    Notes:
    - Uses seaborn for the bars (and your colormap palette).
    - Adds error bars with matplotlib (seaborn yerr is not reliable across versions).
    - Uses *asymmetric* error bars: [distance - lower, upper - distance].
    """

    # import os
    # import numpy as np
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import matplotlib.cm as cm

    # --- distances ---
    distances_from_cntry = dist_matrix.loc[cntry].copy()
    distances_sorted = distances_from_cntry.sort_values()

    # --- CI for the same row, aligned to same order ---
    ci_lower_sorted = ci_lower.loc[cntry][distances_sorted.index].copy()
    ci_upper_sorted = ci_upper.loc[cntry][distances_sorted.index].copy()

    # --- drop self (do this BEFORE colors and yerr so everything stays aligned) ---
    if cntry in distances_sorted.index:
        distances_sorted = distances_sorted.drop(cntry)
        ci_lower_sorted = ci_lower_sorted.drop(cntry)
        ci_upper_sorted = ci_upper_sorted.drop(cntry)

    # --- compute asymmetric error bars ---
    y = pd.to_numeric(distances_sorted, errors="coerce").values
    lo = pd.to_numeric(ci_lower_sorted, errors="coerce").values
    hi = pd.to_numeric(ci_upper_sorted, errors="coerce").values

    # Replace +/-inf with nan
    y = np.where(np.isfinite(y), y, np.nan)
    lo = np.where(np.isfinite(lo), lo, np.nan)
    hi = np.where(np.isfinite(hi), hi, np.nan)

    # yerr must be (2, n): [lower_errs, upper_errs]
    lower_err = y - lo
    upper_err = hi - y
    yerr = np.vstack([lower_err, upper_err])

    # If any yerr entries are negative (shouldn't happen), clamp to 0
    yerr = np.where(np.isfinite(yerr), yerr, np.nan)
    yerr = np.maximum(yerr, 0)

    # --- colors (based on global min/max of the full matrix, like your original) ---
    min_value = dist_matrix.min().min()
    max_value = dist_matrix.max().max()
    bar_colors = [get_color_from_heatmap(val, min_value, max_value, cm.viridis_r) for val in y]

    # --- plot ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bars (no yerr here!)
    sns.barplot(
        x=distances_sorted.index,
        y=y,
        ax=ax,
        palette=bar_colors,
    )

    # Add error bars on the categorical positions 0..n-1
    xpos = np.arange(len(distances_sorted))
    ax.errorbar(
        xpos,
        y,
        yerr=yerr,
        fmt="none",
        capsize=10,
    )

    # Labels/title
    ax.set_xlabel("Country")
    ax.set_ylabel(f"Distance from {cntry}")
    country_name = dict_countrycode2info.get(cntry, [None, cntry])[1]
    ax.set_title(f"Euclidean distances from {country_name} (sorted)")
    ax.set_axisbelow(True)

    # Color x tick labels by language family
    for tick_label in ax.get_xticklabels():
        code = tick_label.get_text()
        if code in dict_countrycode2info:
            langfam = dict_countrycode2info[code][3]
            if langfam in dict_langfam2color:
                tick_label.set_color(dict_langfam2color[langfam])

    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save
    outpath = f"{prefixoutput}_distances_sorted_{cntry}.png"
    fig.savefig(outpath, dpi=1200, bbox_inches="tight")
    plt.show()

    # --- correlation part (unchanged from your original) ---
    english_proficiency_df = pd.DataFrame(english_proficiency).T
    common_countries = english_proficiency_df.index.intersection(dist_matrix.columns)

    english_proficiency_df = english_proficiency_df.loc[common_countries].copy()
    distances = dist_matrix.loc[cntry, common_countries]

    english_proficiency_df["distance"] = distances
    english_proficiency_df["score"] = pd.to_numeric(english_proficiency_df["score"], errors="coerce")
    english_proficiency_df["distance"] = pd.to_numeric(english_proficiency_df["distance"], errors="coerce")

    corr_factor, p_value = plot_eng_correlation(english_proficiency_df, cntry, prefixoutput)
    return corr_factor, p_value

# def create_bar_plot(cntry, dist_matrix, dict_countrycode2info, dict_langfam2color,  ci_lower, ci_upper , prefixoutput):
#      # Calculate distances from USA
#     distances_from_USA = dist_matrix.loc[cntry]
#     print(f"distances_from_USA {distances_from_USA}")
#     # Sort distances in ascending order
#     distances_sorted = distances_from_USA.sort_values()

#     #FINDING COLOR SCHEME
#     # Find the minimum and maximum values of the distance matrix
#     min_value = dist_matrix.min().min()
#     max_value = dist_matrix.max().max()


#     # Create the list of colors for the bars based on the values in distances_sorted
#     bar_colors = [get_color_from_heatmap(val, min_value, max_value, cm.viridis_r) for val in distances_sorted.values]


#     # Extract the lower and upper confidence intervals for distances from USA
#     ci_lower_from_USA = ci_lower.loc[cntry]
#     ci_upper_from_USA = ci_upper.loc[cntry]

#     # Sort the confidence intervals in the same order as the distances
#     ci_lower_sorted = ci_lower_from_USA[distances_sorted.index]
#     ci_upper_sorted = ci_upper_from_USA[distances_sorted.index]

#     # Calculate the error bars (difference between the upper and lower confidence intervals)
#     error_bars = ci_upper_sorted - ci_lower_sorted

#     # Drop the target country from distances_sorted
#     distances_sorted = distances_sorted.drop(cntry)

#     # Drop the target country from the error bars
#     error_bars = error_bars.drop(cntry)

#     # Create a bar plot
#     fig, ax = plt.subplots(figsize=(12, 6))

#     #sns.barplot(x=distances_sorted.index, y=distances_sorted.values, ax=ax, palette=bar_colors, yerr=error_bars)
#     sns.barplot(x=distances_sorted.index, y=distances_sorted.values, ax=ax, palette=bar_colors, yerr=error_bars, capsize=10)

#     # Create a bar plot
#     #fig, ax = plt.subplots(figsize=(12,6))
#     #sns.barplot(x=distances_sorted.index, y=distances_sorted.values, ax=ax)
#     #sns.barplot(x=distances_sorted.index, y=distances_sorted.values, ax=ax, palette=bar_colors)

#     # Add labels and title
#     ax.set_xlabel('Country')
#     ax.set_ylabel(f'Distance from {cntry}')
#     ax.set_title(f'Eucledian distances from {dict_countrycode2info[cntry][1]} (sorted)')

#     for i, tick_label in enumerate(ax.get_xticklabels()):
#             # Get the country code from the x-axis label
#             country_code = tick_label.get_text()
#             if country_code in dict_countrycode2info:
#                 language_family = dict_countrycode2info[country_code][3]
#                 if language_family in dict_langfam2color:
#                     tick_label.set_color(dict_langfam2color[language_family])

#     # Add confidence intervals
#     #n = len(df_mean_imp)
#     #se = stats.sem(distances_sorted)
#     #ci = se * stats.t.ppf((1 + 0.95) / 2, n-1)
#     #ax.axhline(y=distances_sorted.mean(), color='black', linestyle='dotted', label='mean')        
#     #ax.fill_between([0, n-1], distances_sorted.mean()-ci, distances_sorted.mean()+ci, color='gray', alpha=0.3, label='95% CI')
#     #ax.legend()

#     # Rotate x-tick labels for readability
#     plt.xticks(rotation=90)

#     # Save and show the figure
#     plt.savefig(f"{prefixoutput}_distances_sorted_{cntry}.png", dpi=1200)
#     plt.show()

#     # Create a dataframe with English proficiency scores and distances
#     english_proficiency_df = pd.DataFrame(english_proficiency).T
#     #print(english_proficiency_df)
#     # Find the common countries in english_proficiency_df and dist_matrix
#     common_countries = english_proficiency_df.index.intersection(dist_matrix.columns)
#     #print(english_proficiency_df)
#     # Filter english_proficiency_df and dist_matrix to only include common countries
#     english_proficiency_df = english_proficiency_df.loc[common_countries]
#     #print(english_proficiency_df)
#     distances = dist_matrix.loc[cntry, common_countries]

#     # Add the distance column to english_proficiency_df
#     english_proficiency_df['distance'] = distances
#     english_proficiency_df['score']    = pd.to_numeric(english_proficiency_df['score'], errors='coerce')
#     english_proficiency_df['distance'] = pd.to_numeric(english_proficiency_df['distance'], errors='coerce')

#     #this returns  corr_factor, p_value
#     corr_factor, p_value = plot_eng_correlation(english_proficiency_df, cntry ,prefixoutput)
#     return corr_factor, p_value


# In[18]:


def process_country(cntr, dist_matrix, dict_countrycode2info, dict_langfam2color, ci_lower, ci_upper, prefixoutput):
    #try:
    #return corr_factor, p_value       
    print(f"now processing country: {cntr} ")
    corr_factor, p_value = create_bar_plot(cntr, dist_matrix, dict_countrycode2info, dict_langfam2color, ci_lower, ci_upper, prefixoutput)
    print(f"Successfully processed country: {cntr} {corr_factor} {p_value}")

    return corr_factor, p_value
    #except Exception as e:
    #    return f"Error processing country {cntr}: {e}"




# In[19]:


def circular_histogram(data, title, bins, color,filenameoutcirc):
    # Compute the bin edges
    bin_edges = np.linspace(0, 2 * np.pi, bins + 1)

    # Calculate the histogram
    count, bin_edges = np.histogram(data, bin_edges)

    # Normalize the count to the range [0, 1]
    count = count / count.max()

    # Compute the width of each bin
    width = 2 * np.pi / bins

    # Create the circular histogram
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    bars = ax.bar(bin_edges[:-1], count, width=width, bottom=0.0, color=color, alpha=0.5)

    # Set title and format the plot
    plt.title(title)
    ax.set_yticklabels([])
    ax.set_xticks(np.linspace(0, 2 * np.pi, bins, endpoint=False))

    plt.savefig(filenameoutcirc,dpi=1200)
    plt.show()


# In[20]:


def circular_histogram_category(angles, title, bins, filenameoutcirc, categories, questions_to_categories, category_colors):
    unique_categories = set(categories.values())
    for category in unique_categories:
        #filtered_angles = [angle for i, angle in enumerate(angles) if questions_to_categories[i]["category"] == category]
        #color = category_colors[category]
        filtered_angles = [angle for i, angle in enumerate(angles) if questions_to_categories[features[i]]["category"] == category]
        color = category_colors[category]

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        counts, bin_edges = np.histogram(filtered_angles, bins=bins, range=(0, 2 * np.pi))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bars = ax.bar(bin_centers, counts, width=(2 * np.pi / bins), color=color, alpha=0.7)
        ax.set_title(f'{title} - {category}')
        plt.savefig(filenameoutcirc + f'_{category}.png', dpi=1200)
        plt.show()


# In[84]:


# In[21]:


import numpy as np
import pandas as pd
import scipy.stats as stats
from itertools import combinations
import time
import os

def divisive_questions_all_pairs_verbose(
    subset: pd.DataFrame,
    prefixoutput: str,
    alpha: float = 0.05,
    min_n: int = 2,
    verbose: bool = True,
    write_per_pair_rankings: bool = True,
):
    """
    For ALL unordered country pairs:
      1) compute Welch t-test per question
      2) identify most divisive (max |t|) and least different (min |t|, prefer p>=alpha)
      3) write global summary files:
           - prefixoutput_divisive_all_pairs.txt
           - prefixoutput_divisive_all_pairs.tex
      4) additionally, for each pair A_B, write ALL questions ranked by p-value:
           - prefixoutput_A_B_questions.txt
           - prefixoutput_A_B_questions.tex

    Uses global dict_questions for question text.
    """

    # ---------- helpers ----------
    def q_text(q):
        return dict_questions.get(q, {}).get("question", q)

    def q_cat(q):
        return dict_questions.get(q, {}).get("category", "")

    def esc_latex(s: str) -> str:
        s = str(s)
        # minimal safe escaping
        return (s.replace("\\", "\\textbackslash{}")
                 .replace("&", "\\&")
                 .replace("%", "\\%")
                 .replace("$", "\\$")
                 .replace("#", "\\#")
                 .replace("_", "\\_")
                 .replace("{", "\\{")
                 .replace("}", "\\}")
                 .replace("~", "\\textasciitilde{}")
                 .replace("^", "\\textasciicircum{}"))

    def safe_pair_tag(a, b):
        # ensure stable filenames
        return f"{a}_{b}"

    # Put per-pair outputs next to the prefixoutput path
    out_dir = os.path.dirname(prefixoutput)
    if out_dir == "":
        out_dir = "."

    countries = subset["COUNTRIES"].unique()
    qcols = [c for c in subset.columns if c != "COUNTRIES"]
    n_pairs = len(countries) * (len(countries) - 1) // 2

    rows_summary = []
    t0 = time.time()

    if verbose:
        print("[INFO] Starting pairwise divisiveness analysis (all-pairs)")
        print(f"[INFO] Countries: {len(countries)}")
        print(f"[INFO] Questions: {len(qcols)}")
        print(f"[INFO] Total pairs: {n_pairs}")
        print(f"[INFO] Per-pair ranking files: {'ON' if write_per_pair_rankings else 'OFF'}")
        print("-" * 72)

    for idx, (c1, c2) in enumerate(combinations(countries, 2), start=1):
        pair_t0 = time.time()

        if verbose:
            print(f"[{idx:>4}/{n_pairs}] {c1} vs {c2}")

        df2 = subset[subset["COUNTRIES"].isin([c1, c2])]
        A = df2[df2["COUNTRIES"] == c1]
        B = df2[df2["COUNTRIES"] == c2]

        stats_rows = []
        for q in qcols:
            x = pd.to_numeric(A[q], errors="coerce").dropna().to_numpy()
            y = pd.to_numeric(B[q], errors="coerce").dropna().to_numpy()

            if len(x) < min_n or len(y) < min_n:
                continue

            t, p = stats.ttest_ind(x, y, equal_var=False)
            if not np.isfinite(t) or not np.isfinite(p):
                continue

            m1, m2 = float(np.mean(x)), float(np.mean(y))
            diff = m1 - m2

            v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
            pooled = np.sqrt((v1 + v2) / 2) if (v1 + v2) > 0 else np.nan
            d = diff / pooled if (np.isfinite(pooled) and pooled > 0) else np.nan

            stats_rows.append({
                "q": q,
                "text": q_text(q),
                "category": q_cat(q),
                "nA": len(x),
                "nB": len(y),
                "meanA": m1,
                "meanB": m2,
                "diff": diff,
                "t": float(t),
                "abs_t": abs(float(t)),
                "p": float(p),
                "d": float(d) if np.isfinite(d) else np.nan,
            })

        if not stats_rows:
            if verbose:
                print("   ⚠️ no valid questions, skipping\n")
            continue

        dfq = pd.DataFrame(stats_rows)

        # --- summary picks ---
        most = dfq.loc[dfq["abs_t"].idxmax()]
        nonsig = dfq[dfq["p"] >= alpha]
        least = (nonsig.loc[nonsig["abs_t"].idxmin()] if len(nonsig) > 0
                 else dfq.loc[dfq["abs_t"].idxmin()])

        rows_summary.append({
            "A": c1, "B": c2,
            "most_q": most.q, "most_text": most.text, "most_t": most.t, "most_p": most.p, "most_d": most.d,
            "least_q": least.q, "least_text": least.text, "least_t": least.t, "least_p": least.p, "least_d": least.d,
        })

        # --- write per-pair ranking files (ALL questions by p) ---
        if write_per_pair_rankings:
            pair_tag = safe_pair_tag(c1, c2)

            dfq_sorted = dfq.sort_values(["p", "abs_t"], ascending=[True, False]).reset_index(drop=True)

            txt_path = os.path.join(out_dir, f"{os.path.basename(prefixoutput)}_{pair_tag}_questions.txt")
            tex_path = os.path.join(out_dir, f"{os.path.basename(prefixoutput)}_{pair_tag}_questions.tex")

            # TXT
            with open(txt_path, "w") as f:
                f.write(f"All questions ranked by p-value (Welch t-test)\n")
                f.write(f"Pair: {c1} vs {c2}\n")
                f.write(f"min_n per country per question: {min_n}\n\n")
                f.write("Rank\tQ\tp\t|t|\tt\td\tmeanA\tmeanB\tdiff\tnA\tnB\tCategory\tQuestion\n")
                for r_i, r in dfq_sorted.iterrows():
                    f.write(
                        f"{r_i+1}\t{r['q']}\t{r['p']:.6g}\t{r['abs_t']:.6g}\t{r['t']:.6g}\t"
                        f"{r['d'] if np.isfinite(r['d']) else np.nan:.6g}\t"
                        f"{r['meanA']:.6g}\t{r['meanB']:.6g}\t{r['diff']:.6g}\t"
                        f"{int(r['nA'])}\t{int(r['nB'])}\t{r['category']}\t{r['text']}\n"
                    )

            # LaTeX (longtable so it can span pages)
            with open(tex_path, "w") as f:
                f.write("\\begin{longtable}{r l r r r r r r r r r p{2.5cm} p{7cm}}\n")
                f.write(f"\\caption{{All questions ranked by p-value (Welch t-test): {esc_latex(c1)} vs {esc_latex(c2)}}}\\\\\n")
                f.write("\\toprule\n")
                f.write("Rank & Q & $p$ & $|t|$ & $t$ & $d$ & $\\bar{x}_A$ & $\\bar{x}_B$ & diff & $n_A$ & $n_B$ & Category & Question\\\\\n")
                f.write("\\midrule\n\\endfirsthead\n")
                f.write("\\toprule\n")
                f.write("Rank & Q & $p$ & $|t|$ & $t$ & $d$ & $\\bar{x}_A$ & $\\bar{x}_B$ & diff & $n_A$ & $n_B$ & Category & Question\\\\\n")
                f.write("\\midrule\n\\endhead\n")
                f.write("\\bottomrule\n\\endfoot\n")

                for r_i, r in dfq_sorted.iterrows():
                    dval = r["d"] if np.isfinite(r["d"]) else np.nan
                    f.write(
                        f"{r_i+1} & {esc_latex(r['q'])} & {r['p']:.3g} & {r['abs_t']:.2f} & {r['t']:.2f} & "
                        f"{dval:.2f} & {r['meanA']:.2f} & {r['meanB']:.2f} & {r['diff']:.2f} & "
                        f"{int(r['nA'])} & {int(r['nB'])} & {esc_latex(r['category'])} & {esc_latex(r['text'])}\\\\\n"
                    )

                f.write("\\end{longtable}\n")

        # --- verbose progress ---
        if verbose:
            elapsed = time.time() - t0
            eta = (elapsed / idx) * (n_pairs - idx)
            pair_dt = time.time() - pair_t0

            print(
                f"   ✔ usable questions: {len(dfq)} | wrote ranking: {('yes' if write_per_pair_rankings else 'no')}\n"
                f"     MOST  : {most.q} – {most.text}\n"
                f"             |t|={most.abs_t:.2f}, p={most.p:.2g}, d={most.d:.2f}\n"
                f"     LEAST : {least.q} – {least.text}\n"
                f"             |t|={abs(least.t):.2f}, p={least.p:.2g}\n"
                f"   ⏱ pair {pair_dt:.2f}s | elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min\n"
            )

    result_df = pd.DataFrame(rows_summary)

    # --- global summary files ---
    txtfile = prefixoutput + "_divisive_all_pairs.txt"
    texfile = prefixoutput + "_divisive_all_pairs.tex"

    with open(txtfile, "w") as f:
        for _, r in result_df.iterrows():
            f.write(f"{r.A} vs {r.B}\n")
            f.write(f"  MOST : {r.most_q}  | t={r.most_t:.3g}, p={r.most_p:.3g}, d={r.most_d:.3g}\n")
            f.write(f"         {r.most_text}\n")
            f.write(f"  LEAST: {r.least_q} | t={r.least_t:.3g}, p={r.least_p:.3g}, d={r.least_d:.3g}\n")
            f.write(f"         {r.least_text}\n\n")

    def esc(s): return str(s).replace("_", "\\_").replace("&", "\\&")

    with open(texfile, "w") as f:
        f.write("\\begin{tabular}{llp{6cm}rrp{6cm}rr}\n")
        f.write("\\toprule\n")
        f.write("A & B & Most divisive & $t$ & $p$ & Least different & $t$ & $p$ \\\\\n")
        f.write("\\midrule\n")
        for _, r in result_df.iterrows():
            f.write(
                f"{esc(r.A)} & {esc(r.B)} & {esc(r.most_text)} & {r.most_t:.2f} & {r.most_p:.2g} & "
                f"{esc(r.least_text)} & {r.least_t:.2f} & {r.least_p:.2g} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")

    if verbose:
        print("-" * 72)
        print(f"[DONE] Wrote summary: {txtfile}")
        print(f"[DONE] Wrote summary: {texfile}")
        if write_per_pair_rankings:
            print(f"[DONE] Wrote per-pair ranked files like: {os.path.basename(prefixoutput)}_ARG_JOR_questions.txt/.tex")

    return result_df


# In[22]:


import numpy as np
import pandas as pd

def country_variance_extremes(subset: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    """
    For each country, find the question with:
      - highest within-country variance
      - lowest within-country variance (among questions with >=min_n responses)

    subset must contain column 'COUNTRIES' and question columns (numeric or NaN).
    Returns a DataFrame with one row per country.
    """
    def q_text(q):
        return dict_questions.get(q, {}).get("question", q)

    qcols = [c for c in subset.columns if c != "COUNTRIES"]
    rows = []

    for country, g in subset.groupby("COUNTRIES"):
        # compute variance per question using only available answers
        variances = {}
        counts = {}

        for q in qcols:
            x = pd.to_numeric(g[q], errors="coerce").dropna()
            counts[q] = int(x.shape[0])
            if x.shape[0] >= min_n:
                variances[q] = float(x.var(ddof=1))
            else:
                variances[q] = np.nan

        s_var = pd.Series(variances)
        s_n = pd.Series(counts)

        valid = s_var.dropna()
        if valid.empty:
            # no questions passed min_n
            rows.append({
                "country": country,
                "highest_var_q": None, "highest_var_text": None, "highest_var": np.nan, "highest_var_n": 0,
                "lowest_var_q": None, "lowest_var_text": None, "lowest_var": np.nan, "lowest_var_n": 0,
            })
            continue

        q_high = valid.idxmax()
        q_low  = valid.idxmin()

        rows.append({
            "country": country,

            "highest_var_q": q_high,
            "highest_var_text": q_text(q_high),
            "highest_var": float(valid.loc[q_high]),
            "highest_var_n": int(s_n.loc[q_high]),

            "lowest_var_q": q_low,
            "lowest_var_text": q_text(q_low),
            "lowest_var": float(valid.loc[q_low]),
            "lowest_var_n": int(s_n.loc[q_low]),
        })

    return pd.DataFrame(rows).sort_values("country").reset_index(drop=True)


def write_country_variance_extremes(df_ext: pd.DataFrame, prefixoutput: str):
    """
    Writes TXT + LaTeX summaries for country variance extremes.
    """
    def esc(s):
        return str(s).replace("_", "\\_").replace("&", "\\&")

    txtfile = prefixoutput + "_country_variance_extremes.txt"
    texfile = prefixoutput + "_country_variance_extremes.tex"

    # TXT
    with open(txtfile, "w") as f:
        for _, r in df_ext.iterrows():
            f.write(f"{r['country']}\n")
            f.write(f"  HIGHEST variance:\n")
            f.write(f"    {r['highest_var_q']}: {r['highest_var_text']}\n")
            f.write(f"    var={r['highest_var']:.4g}, n={r['highest_var_n']}\n")
            f.write(f"  LOWEST variance:\n")
            f.write(f"    {r['lowest_var_q']}: {r['lowest_var_text']}\n")
            f.write(f"    var={r['lowest_var']:.4g}, n={r['lowest_var_n']}\n\n")

    # LaTeX
    with open(texfile, "w") as f:
        f.write("\\begin{tabular}{l p{6cm} r r p{6cm} r r}\n")
        f.write("\\toprule\n")
        f.write("Country & Highest-variance question & var & $n$ & Lowest-variance question & var & $n$ \\\\\n")
        f.write("\\midrule\n")
        for _, r in df_ext.iterrows():
            f.write(
                f"{esc(r['country'])} & {esc(r['highest_var_text'])} & {r['highest_var']:.3g} & {int(r['highest_var_n'])} & "
                f"{esc(r['lowest_var_text'])} & {r['lowest_var']:.3g} & {int(r['lowest_var_n'])} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")

    print("Wrote:", txtfile)
    print("Wrote:", texfile)


# In[23]:


#df is the data frame
#prefixoutput
#d is debug or not
def performAnalyses(df,prefixoutput, d):  
    #-2 means No answer, -4 means Question was not asked in this country, -5 means Missing; Not available

    #@MB can you please double check that the filtering that we are doing is robust to this filtering to 80%?
    print("performAnalyses"+str(prefixoutput))

    #Delete from our DataFrame the columns where the proportion of rows with value -4 is greater than 0.8
    #(Where the question has not been asked in more than 80% of countries)
    count_minus_four = df.apply(lambda x: x[x == -4].count(), axis=0)
    prop_minus_four = count_minus_four / len(df)
    columns_to_delete = prop_minus_four[prop_minus_four > 0.8].index
    df = df.drop(columns_to_delete, axis=1)
    print("1 "+str(df.shape))

    print("2 "+str(df.shape))
    #keep ONLY the questions
    df1 = df.loc[:, df.columns.str.startswith(('Q', 'X', 'V'))]
    df1=df1.drop(['Q_MODE', 'VERSION'], axis=1)
    #print("shape")
    #print(df1.shape)
    #print("col")
    column_names = list(df1.columns)
    #print(column_names)

    #Filtering values-based questions
    # Create an array with the indices of columns to drop using slice notation
    #cols_to_drop = np.r_[52:59, 94:97, 146:164, 215:252, 241:254, 270:279, 327:377]
    # Filter cols_to_drop to keep only valid indices within the range of DataFrame's columns
    #cols_to_drop = cols_to_drop[cols_to_drop < len(df1.columns)]

    # Print the column names corresponding to the indices in cols_to_drop
    #column_names_to_drop = df1.columns[cols_to_drop]
    #print(column_names_to_drop.tolist())

    # Filter the list to keep only columns that are present in the DataFrame
    #cols_to_drop = [col for col in cols_to_remove if col in df1.columns]
    #print(cols_to_drop)
    #return;
    # Get the indices of the columns to remove and concatenate them with the existing cols_to_drop array
    #cols_to_drop = np.concatenate([cols_to_drop, [df1.columns.get_loc(col) for col in cols_to_rm]])

    # Filter cols_to_drop to keep only valid indices within the range of DataFrame's columns
    #cols_to_drop = cols_to_drop[cols_to_drop < len(df1.columns)]

    # Check and collect columns that exist in the DataFrame
    cols_to_remove_existing = []
    for col in cols_to_remove:
        if col in df1.columns:
            cols_to_remove_existing.append(col)
            #print(f"Column '{col}' exists in the DataFrame and will be removed.")
        #else:
        #    print(f"Column '{col}' does not exist in the DataFrame.")

    # Remove the existing columns from the DataFrame
    df1 = df1.drop(cols_to_remove_existing, axis=1)
    # Drop the columns
    #df1 = df1.drop(df1.columns[cols_to_drop], axis=1)

    #    cols_to_drop = np.r_[52:59, 94:97, 146:164, 215:252, 241:254, 270:279, 327:377, [df1.columns.get_loc(col) for col in ['Q112', 'Q118', 'Q120', 'Q234A','Q82_EU']]]
    #    df1 = df1.drop(df1.columns[cols_to_drop], axis=1)
    df = df.loc[:, ['COUNTRIES', 'K_DURATION']]
    #DataFrame containing only the countries, K_Duration column and the Questions
    df=pd.concat([df.iloc[:, :2], df1, df.iloc[:, 2:]], axis=1)

    print("3"+str(df.shape))


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



    ###################################################################
    #random forest
    ################################################################
#    if d==0:
    if True:    
        print(subset.columns.tolist())
        #Create a new DataFrame with the same content as subset

        # Create a new DataFrame with the same content as subset
        subset_rf = subset.copy()

        # Create a new column in the DataFrame with the language family for each country
        subset_rf['LANGUAGE_FAMILY'] = subset_rf['COUNTRIES'].apply(lambda x: dict_countrycode2info[x][3])

        # Drop the COUNTRIES column
        subset_rf.drop('COUNTRIES', axis=1, inplace=True)

        # One-hot encode categorical columns (replace 'COLUMN_NAME_1', 'COLUMN_NAME_2', etc. with your categorical column names)
        #subset_rf = pd.get_dummies(subset_rf, columns=['COLUMN_NAME_1', 'COLUMN_NAME_2', ...])

        # Encode the LANGUAGE_FAMILY column as integers
        language_family_encoder = LabelEncoder()
        subset_rf['ENCODED_LANGUAGE_FAMILY'] = language_family_encoder.fit_transform(subset_rf['LANGUAGE_FAMILY'])
        #print( np.unique(subset_rf['ENCODED_LANGUAGE_FAMILY']))
        #return subset,subset,subset

        # Get a list of unique language families
        enc_language_families = subset_rf['ENCODED_LANGUAGE_FAMILY'].unique()

        # Drop the LANGUAGE_FAMILY columns
        subset_rf.drop(['LANGUAGE_FAMILY'], axis=1, inplace=True)

        # Replace missing and infinite values with the mean for each language family in that column
        for col in subset_rf.columns:
            if col != 'COUNTRIES':
                for language_family in enc_language_families:
                    mask = subset_rf['ENCODED_LANGUAGE_FAMILY'] == language_family
                    mean_value = subset_rf.loc[mask, col].replace([np.inf, -np.inf], np.nan).mean(skipna=True)
                    subset_rf.loc[mask, col] = subset_rf.loc[mask, col].replace([np.nan, np.inf, -np.inf], mean_value)
        print(subset.head())
        #global_means = subset.mean()
        global_means = subset.select_dtypes(include="number").mean()


        # Create an empty list to store the rows of the LaTeX table
        latex_rows = []

        # Open a new text file to write the top 5 features
        with open(prefixoutput+'top_5_features_per_language_family.txt', 'w') as f:
            for language_family in enc_language_families:
                language_family_name = language_family_encoder.inverse_transform([language_family])[0]
                print(f"Training classifier for language family: {language_family_name}")
                f.write(f"Training classifier for language family: {language_family_name}\n")

                # Create binary labels for the current language family
                y_binary = (subset_rf['ENCODED_LANGUAGE_FAMILY'] == language_family).astype(int)

                # Split the data into training and testing sets
                X = subset_rf.drop('ENCODED_LANGUAGE_FAMILY', axis=1)
                y = y_binary
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the Random Forest model
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)

                # Make predictions on the test set
                y_pred = clf.predict(X_test)

                # Calculate the number of incorrect predictions
                num_incorrect = np.sum(y_pred != y_test)

                # Print the number of incorrect predictions
                print(f"The classifier got {num_incorrect} predictions wrong.")
                f.write(f"The classifier got {num_incorrect} predictions wrong.\n")

                # Calculate the accuracy and print it
                accuracy = accuracy_score(y_test, y_pred)
                print(f"The classifier's accuracy is {accuracy:.2f}.")
                f.write(f"The classifier's accuracy is {accuracy:.2f}.\n")

                # Get feature importances
                importances = clf.feature_importances_

                # Get the top 5 features and their importances
                top_5_features_indices = np.argsort(importances)[-5:]
                top_5_features = X.columns[top_5_features_indices][::-1]
                top_5_importances = importances[top_5_features_indices][::-1]

                # Print and write the top 5 features and their importances
                for feature, importance in zip(top_5_features, top_5_importances):
                    question = dict_questions[feature]["question"] if feature in dict_questions else feature
                    print(f"Feature: {question}, Importance: {importance:.2f}")
                    f.write(f"Feature: {question}, Importance: {importance:.2f}\n")

                    # Calculate the mean for the current language family for this feature
                    language_family_mean = subset_rf.loc[subset_rf['ENCODED_LANGUAGE_FAMILY'] == language_family, feature].mean()

                    # Get the global mean for this feature
                    global_mean = global_means[feature]

                    # Print and write the comparison
                    print(f"Mean for {language_family_name}: {language_family_mean:.2f}, Global mean: {global_mean:.2f}")
                    f.write(f"Mean for {language_family_name}: {language_family_mean:.2f}, Global mean: {global_mean:.2f}\n")

                    # Append a row to the LaTeX table
                    latex_rows.append([language_family_name, question, importance, language_family_mean, global_mean])

                print("\n")
                f.write("\n")



        # Print the LaTeX table
        print(tabulate(latex_rows, headers=["Language Family", "Question", "Importance", "Language Family Mean", "Global Mean"], tablefmt="latex"))

        # Write the LaTeX table to a file
        with open(prefixoutput+'top_5_features_per_language_family.tex', 'w') as f:
            f.write(tabulate(latex_rows, headers=["Language Family", "Question", "Importance", "Language Family Mean", "Global Mean"], tablefmt="latex"))

    #pairwise division
    pairwise_divisiveness = divisive_questions_all_pairs_verbose(
                                                                subset,
                                                                prefixoutput,
                                                                #dict_questions=dict_questions,
                                                                verbose=True,
    )

    #computing the most divisive question
    df_country_var_ext = country_variance_extremes(subset, min_n=10)
    write_country_variance_extremes(df_country_var_ext, prefixoutput)

    ############################
    # computing means

    #computing the mean per country
    df_means=subset.groupby(['COUNTRIES']).mean()
    df_means_cp=subset.groupby(['COUNTRIES'], as_index=False).mean()

    #computing the variance per country
    df_variance=subset.groupby(['COUNTRIES']).var()

    #we are computing the heatmap to see the missing values
    #dfmeans_heatmap=plot_heatmap(df_means)
    #dfmeans_heatmap.savefig(prefix+"_heatmap.png")


    #
    # IMPUTATION using KNN Imputer
    #    
    imputer = KNNImputer(n_neighbors=2)
    df_mean_imp = pd.DataFrame(imputer.fit_transform(df_means),columns=df_means.columns)
    df_mean_imp = df_mean_imp.set_index(df_means.index)

    df_variance_imp = pd.DataFrame(imputer.fit_transform(df_variance),columns=df_variance.columns)
    df_variance_imp = df_variance_imp.set_index(df_variance.index)
  #  return ;

#    if d==0:
    if True:


        font_path = fm.findfont(fm.FontProperties(family='monospace'))
        font_name = fm.FontProperties(fname=font_path).get_name()        

        ################################################################

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

        # Generate the tick label colors
        #         tick_label_colors = [dict_langfam2color[language_families[code]] for code in df_mean_imp.index]
        #         Qtick_label_colors = [dict_category2color[dict_questions[question]["category"]] for question in df_mean_imp.columns]

        #         # Create the heatmap plot
        #         plt.figure(figsize=(40, 19))
        #         a = sns.heatmap(df_mean_imp, vmin=0, vmax=5, square=True, cmap="viridis_r", linewidths=0.1,
        #                         annot=True, annot_kws={"fontsize": 3.5}, xticklabels=1, yticklabels=1)
        #         a.set(xlabel="QUESTIONS", ylabel="COUNTRIES")

        #         # Set the fixed-width font for y-axis and x-axis tick labels
        #         for i, tick_label in enumerate(a.axes.get_yticklabels()):
        #             tick_label.set_color(tick_label_colors[i])
        #             tick_label.set_fontproperties(fm.FontProperties(fname=font_path, size=8)) 
        #             tick_label.set_fontweight('bold')

        #         for i, tick_label in enumerate(a.axes.get_xticklabels()):
        #             tick_label.set_color(Qtick_label_colors[i])
        #             tick_label.set_fontproperties(fm.FontProperties(fname=font_path, size=8))
        #             tick_label.set_text(tick_label.get_text().ljust(5))            
        #             tick_label.set_fontweight('bold')

        #         plt.savefig(prefixoutput+"_heatmap_imp.png")
        #         plt.show()


        ###############################################################
        #
        # THIS IS THE RAW HEATMAP
        #
        #         tick_label_colors = [dict_langfam2color[language_families[code]] for code in df_mean_imp.index]
        #         plt.figure(figsize = (40,19))
        #         heatmap = sns.heatmap(df_means, vmin=0, vmax=5, square=True, cmap="viridis_r", linewidths=0.1, 
        #                 annot=True, annot_kws={"fontsize":3.5},
        #                  xticklabels=1, yticklabels=1)
        #         heatmap.set(xlabel="QUESTIONS", ylabel="COUNTRIES")
        #         plt.savefig(prefixoutput+"_rawheatmap.png")
        #         for i, tick_label in enumerate(heatmap.axes.get_yticklabels()):
        #            tick_label.set_color(tick_label_colors[i])
        #         plt.show()
        # Create the heatmap plot
        plt.figure(figsize=(40, 19))
        a = sns.heatmap(df_means, vmin=0, vmax=5, square=True, cmap="viridis_r", linewidths=0.1,
                        annot=True, annot_kws={"fontsize": 3.5}, xticklabels=1, yticklabels=1)
        a.set(xlabel="QUESTIONS", ylabel="COUNTRIES")

        # Set the fixed-width font for y-axis and x-axis tick labels
        for i, tick_label in enumerate(a.get_yticklabels()):
            tick_label.set_fontproperties(fm.FontProperties(fname=font_path, size=8))
            country_code = tick_label.get_text()
            if country_code in dict_countrycode2info:
                language_family = dict_countrycode2info[country_code][3]
                if language_family in dict_langfam2color:
                    tick_label.set_color(dict_langfam2color[language_family])
            tick_label.set_rotation(0)  # To make labels horizontal 

        for i, tick_label in enumerate(a.get_xticklabels()):
            tick_label.set_fontproperties(fm.FontProperties(fname=font_path, size=8))
            question_code = tick_label.get_text()
            if question_code in dict_questions:
                category = dict_questions[question_code]["category"]
                if category in dict_category2color:
                    color = dict_category2color[category]
                    tick_label.set_color(color)
            tick_label.set_rotation(90)  # To make labels vertical


        plt.savefig(prefixoutput+"_heatmap.png") 
        caption = "HEATMAP."  
        plt.figtext(0.5, 0.01, caption, ha="center", fontsize=10)
        plt.show()


        subfile_name = prefixoutput + '_heatmap.tex'

        # Generate the subfile
        with open(subfile_name, 'w') as f:
            f.write('\\begin{document}\n')
            f.write('\\begin{figure}[htbp]\n')
            f.write('\\centering\n')
            f.write('\\includegraphics[width=\\linewidth]{%s_heatmap.png}\n' % prefixoutput)
            f.write('\\caption{My heatmap for.}\n')
            f.write('\\label{fig:my_heatmap}\n')
            f.write('\\end{figure}\n')


        #return ;
        ################################################################
        #
        # CLUSTERMAP WITH IMPUTED VALUES
        #
        #@MB add the colors of the countries to the labels              xxxxDONE
        plt.figure(figsize = (40,19))
        g = sns.clustermap(df_mean_imp,figsize=(40, 19), dendrogram_ratio=(.1, .2),vmin=0, vmax=5, cmap="viridis_r", linewidths=0.1, annot=True, annot_kws={"fontsize":4}, xticklabels=1, yticklabels=1)
        for i, tick_label in enumerate(g.ax_heatmap.get_ymajorticklabels()):
            tick_label.set_fontproperties(fm.FontProperties(fname=font_path, size=10))
            country_code = tick_label.get_text()
            if country_code in dict_countrycode2info:
                language_family = dict_countrycode2info[country_code][3]
                if language_family in dict_langfam2color:
                    tick_label.set_color(dict_langfam2color[language_family])
            tick_label.set_label("COUNTRIES") 

        for i, tick_label in enumerate(g.ax_heatmap.get_xmajorticklabels()):
            tick_label.set_fontproperties(fm.FontProperties(fname=font_path, size=10))
            question_code = tick_label.get_text()
            if question_code in dict_questions:
                category = dict_questions[question_code]["category"]
                if category in dict_category2color:
                    color = dict_category2color[category]
                    tick_label.set_color(color)
            tick_label.set_label("QUESTIONS")
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), rotation=0)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), rotation=90)
        plt.savefig(prefixoutput+"_clustermap_imp.png", dpi=300)
        plt.show()

        subfile_name = prefixoutput + '_heatmap_imp.tex'

        # Generate the subfile
        with open(subfile_name, 'w') as f:
            f.write('\\begin{document}\n')
            f.write('\\begin{figure}[htbp]\n')
            f.write('\\centering\n')
            f.write('\\includegraphics[width=\\linewidth]{%s_heatmap_imp.png}\n' % prefixoutput)
            f.write('\\caption{My heatmap.}\n')
            f.write('\\label{fig:my_heatmap\n')
            f.write('\\end{figure}\n')


        ################################################################

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
        x_axis_categories = [dict_questions[q]["category"] for q in df_mean_imp.columns[:-1]]
        x_axis_colors = [dict_category2color[cat] for cat in x_axis_categories]
        x_axis_pal = sns.color_palette(x_axis_colors)

        # Create the clustermap with row colors and x-axis colors
        plt.figure(figsize=(30, 37), dpi=600)
        g = sns.clustermap(df_mean_imp.iloc[:, :-1], row_colors=row_colors, cmap="viridis_r", yticklabels=1, col_colors=x_axis_pal)

        # Color x-axis labels based on categories
        for tick_label in g.ax_heatmap.axes.get_xticklabels():
            tick_label.set_fontproperties(fm.FontProperties(fname=font_path, size=10))
            tick_text = tick_label.get_text()
            category = dict_questions[tick_text]["category"]
            tick_label.set_color(dict_category2color[category])
            tick_label.set_label("QUESTIONS") # Set x-axis label as "QUESTIONS"
        # Color y-axis labels based on language family
        for tick_label in g.ax_heatmap.axes.get_yticklabels():
            tick_label.set_fontproperties(fm.FontProperties(fname=font_path, size=8))
            tick_text = tick_label.get_text()
            language_fam_name = df_mean_imp.loc[tick_text, "language_family"]
            tick_label.set_color(lut[language_fam_name])
            tick_label.set_label("COUNTRIES") # Set y-axis label as "COUNTRIES"
        plt.savefig(prefixoutput + "_clustermaplang_imp.png", dpi=300)
        plt.show()






        ################################################################

        #
        # PCA WITH IMPUTED VALUES
        #
        # remove last column which is language family
        dict_country2color = {}
        for country, info in dict_countrycode2info.items():
            lang_fam = info[3]
            color = dict_langfam2color[lang_fam]
            dict_country2color[country] = color

        # Prepare the data
        X = df_mean_imp.iloc[:, :-1]

        # Standardize data
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # Run PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_std)

        # Create a figure
        fig = plt.figure(figsize=(10, 8))

        # Plot scatter with country colors
        colors = [dict_country2color.get(country, (0, 0, 0)) for country in df_mean_imp.index]
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors)

        # Plot labels for data points
        plt.xlabel('PC1', fontproperties=fm.FontProperties(fname=font_path, size=12))
        plt.ylabel('PC2', fontproperties=fm.FontProperties(fname=font_path, size=12))
        for i, label in enumerate(df_mean_imp.index):
            x = X_pca[i, 0]
            y = X_pca[i, 1]
            plt.annotate(label, (x, y), fontproperties=fm.FontProperties(fname=font_path, size=10))

        # Add legends for country color codes
        handles = []
        labels = []
        for lang_fam, color in dict_langfam2color.items():
            handle = plt.scatter([], [], c=[color], label=lang_fam)
            handles.append(handle)
            labels.append(lang_fam)
        plt.legend(handles, labels, fontsize='small')

        # Add percentage of explained variance for PC1 and PC2
        variance_ratios = pca.explained_variance_ratio_
        plt.title(f'PCA ({variance_ratios[0]*100:.2f}% variance explained by PC1, {variance_ratios[1]*100:.2f}% by PC2)', fontproperties=fm.FontProperties(fname=font_path, size=12))

        # Save and show the figure
        plt.savefig(prefixoutput+"_pca_pc1_pc2_imp.png", dpi=1200)
        plt.show()


        #INDIVIDUAL PCAS FOR EACH LANGUAGE FAMILY
        unique_lang_fams = list(set([info[3] for info in dict_countrycode2info.values()]))

        # Prepare the data
        X = df_mean_imp.iloc[:, :-1]

        # Standardize data
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # Create a figure for each language family
        for lang_fam in unique_lang_fams:
            # Get the countries in this language family
            countries = [country for country, info in dict_countrycode2info.items() if info[3] == lang_fam]

            # Create a dictionary mapping countries to their colors
            country_colors = {}
            for country in countries:
                color = dict_country2color[country]
                country_colors[country] = color

            # Run PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_std)

            # Create a figure
            fig = plt.figure(figsize=(10, 8))

            # Plot scatter with country colors
            colors = [country_colors.get(country, (0.5, 0.5, 0.5)) for country in df_mean_imp.index]
            sizes = [120 if country in countries else 15 for country in df_mean_imp.index]
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=sizes)

            # Plot labels for data points
            plt.xlabel('PC1', fontproperties=fm.FontProperties(fname=font_path, size=12))
            plt.ylabel('PC2', fontproperties=fm.FontProperties(fname=font_path, size=12))
            for i, label in enumerate(df_mean_imp.index):
                x = X_pca[i, 0]
                y = X_pca[i, 1]
                plt.annotate(label, (x, y), fontproperties=fm.FontProperties(fname=font_path, size=10))

            # Add legends for country color codes
            handles = []
            labels = []
            for fam in unique_lang_fams:
                color = dict_langfam2color[fam]
                handle = plt.scatter([], [], c=[color], label=fam)
                handles.append(handle)
                labels.append(fam)
            plt.legend(handles, labels, fontsize='small')

            # Add percentage of explained variance for PC1 and PC2
            variance_ratios = pca.explained_variance_ratio_
            plt.title(f'PCA for {lang_fam} ({variance_ratios[0]*100:.2f}% variance explained by PC1, {variance_ratios[1]*100:.2f}% by PC2)', fontproperties=fm.FontProperties(fname=font_path, size=12))

            # Save and show the figure
            plt.savefig(prefixoutput+f"_pca_{lang_fam}_pc1_pc2_imp.png", dpi=1200)
            plt.show()


        #BIPLOT FOR PCA    
        features = X.columns
        scale = 4  # Controls the size of the dots for loading vectors
        fig = plt.figure(figsize=(10, 8))

        # Define the color for each feature based on its category
        colors = []
        for feature in features:
            category = dict_questions[feature]["category"]
            color = dict_category2color[category]
            colors.append(color)

            # Sort components by their explained variance ratio    xxDONE
            #sorted_indices = np.argsort(pca.explained_variance_ratio_)[::-1]    xxDONE

            # Function to determine if a component is in the top 5 for a given direction
        def is_top_5_component(index, direction):
            if direction == 'positive_x':
                return pca.components_[0, index] >= -np.partition(-pca.components_[0, :], 5)[5]
            elif direction == 'negative_x':
                return pca.components_[0, index] <= np.partition(pca.components_[0, :], 5)[5]
            elif direction == 'positive_y':
                return pca.components_[1, index] >= -np.partition(-pca.components_[1, :], 5)[5]
            elif direction == 'negative_y':
                return pca.components_[1, index] <= np.partition(pca.components_[1, :], 5)[5]

        for i, feature in enumerate(features):
            dot_size = scale  # Default dot size for non-top components
            dot_color = colors[i]  # Color based on feature category

            if is_top_5_component(i, 'positive_x') or is_top_5_component(i, 'negative_x') or is_top_5_component(i, 'positive_y') or is_top_5_component(i, 'negative_y'):
                dot_size = scale * 5
                dot_color = 'red'
                if dot_color != dict_category2color[dict_questions[feature]["category"]]: # check if color needs to be changed
                    dot_color = dict_category2color[dict_questions[feature]["category"]] # change color based on category
                    plt.text(scale * pca.components_[0, i], scale * pca.components_[1, i], feature, fontsize=8, ha='center', va='center')
                else:
                    plt.text(scale * pca.components_[0, i], scale * pca.components_[1, i], feature, fontsize=8, ha='center', va='center')

            plt.scatter(scale * pca.components_[0, i], scale * pca.components_[1, i], s=dot_size, marker='o', color=dot_color, alpha=0.7)

        # Adjust plot limits to include loading vectors
        loadings_max = max(scale * pca.components_.max(), -scale * pca.components_.min())
        plt.xlim(-loadings_max * 1.1, loadings_max * 1.1)
        plt.ylim(-loadings_max * 1.1, loadings_max * 1.1)
        plt.title(f'PCA Biplot', fontproperties=fm.FontProperties(fname=font_path, size=12))

        # Create a legend for the color categories
        patches = [mpatches.Patch(color=color, label=category) for category, color in dict_category2color.items()]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save and show the figure
        plt.savefig(prefixoutput+"BiplotPCA.png", dpi=300)
        plt.show()


        #circular_histogram(angles, 'Circular Histogram', 16, 'green', prefixoutput+"_biplotPCA_circ.png")
        angles=[]
        for i, feature in enumerate(features):
            # Calculate the angle of the vector in radians
            angle = np.arctan2(pca.components_[1, i], pca.components_[0, i])

            # Convert the angle to the range [0, 2π]
            if angle < 0:
                angle += 2 * np.pi
            angles.append(angle)  # Append the angle value to the list

        print(angles);
        circular_histogram(angles, 'Circular Histogram all features', 16, 'green', prefixoutput+"_biplotPCA_circ.png")

        unique_categories = set(dict_questions[feature]["category"] for feature in features)

        for category in unique_categories:
            # Filter angles and colors based on the current category
            filtered_angles = [angle for i, angle in enumerate(angles) if dict_questions[features[i]]["category"] == category]
            color = dict_category2color[category]

            # Call the circular_histogram function with the filtered angles and category color
            circular_histogram(filtered_angles, f'Circular Histogram for {category}', 16, color, f'{prefixoutput}_biplotPCA_circ_{category}.png')




        ################################################################

        #
        #  t-SNE WITH IMPUTED VALUES
        # @MB add a set of dots/ country color 
        #like this: https://s3-eu-west-1.amazonaws.com/ppreviews-plos-725668748/980809/preview.jpg     xxxDONE

        # remove last column which is language family
        plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()

        # Rest of the code
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
        plt.savefig(prefixoutput+"_tsne_imp.png", dpi=300)
        plt.show()

















        ################################################################


        #@MB fix the color of countries in nmf
        #@MB using the dictionary of questions, write more representative questions to text file with "prefix"_NMF_#comp_compX_questions.txt, 
        #
        # Non Negative Matrix WITH IMPUTED VALUES
        #
        def plot_triangle(points, labels, colors=None, title=None, filename=None):
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_aspect("equal")

            # Define triangle vertices
            vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])

            # Plot triangle
            ax.plot([vertices[0, 0], vertices[1, 0]], [vertices[0, 1], vertices[1, 1]], 'k-')
            ax.plot([vertices[1, 0], vertices[2, 0]], [vertices[1, 1], vertices[2, 1]], 'k-')
            ax.plot([vertices[2, 0], vertices[0, 0]], [vertices[2, 1], vertices[0, 1]], 'k-')

            # Convert barycentric coordinates to Cartesian coordinates
            cart_coords = points.dot(vertices)

            # Plot points and labels
            for i, (coord, label) in enumerate(zip(cart_coords, labels)):
                point_color = colors[i] if colors is not None else 'k'
                ax.scatter(coord[0], coord[1], color=point_color, marker='o')
                ax.annotate(label, xy=coord, xytext=(5, 0), textcoords="offset points", fontsize=8, color=point_color)

            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.axis("off")

            if title is not None:
                plt.title(title)

            if filename is not None:
                plt.savefig(filename, dpi=1200)

            plt.show()


        for n_components in range(2, 5):
            tick_label_colors = [dict_langfam2color[language_families[code]] for code in df_mean_imp.index]
            nmf_model = NMF(n_components=n_components)

            # remove last column which is language family
            X = df_mean_imp.iloc[:, :-1]

            # Fit NMF model to data
            W = nmf_model.fit_transform(X)
            H = nmf_model.components_

            # Print the top features for each component
            feature_names = df_mean_imp.columns.values
            latex_table = "\\begin{tabular}{c|l}\n"
            latex_table += "Component & Top Features\\\\\\hline\n"

            for i, component in enumerate(H):
                top_features = [dict_questions[feature_names[j]]['question'] for j in component.argsort()[:-10:-1]]
                top_features_str = f"Top features for component {i}:\n" + "\n".join(top_features)
                print(top_features_str)

                # Save the top features to separate text files
                outfile = os.path.join(output_dir, f"NMF_{n_components}_comp_{i}_top_features.txt")

                with open(outfile, "w") as f:
                    f.write(top_features_str + "\n")
                # Add top features to the LaTeX table
                latex_table += f"{i} & "
                latex_table += ", ".join([f"\\textit{{{feature}}}" for feature in top_features])
                latex_table += "\\\\\n"

            latex_table += "\\end{tabular}\n"

            print("LaTeX Tabular Output:")
            print(latex_table)

            # Save the LaTeX table to a file
            latex_outfile = os.path.join(output_dir, f"NMF_{n_components}_latex_tabular_output.tex")
            with open(latex_outfile, "w") as f:
                f.write(latex_table)


            print(W)
            print(H)
            W_norm = W / W.sum(axis=1, keepdims=True)
            print(W_norm)

            # Sort the stacked barplot by the values in the first column of W
            if n_components == 3:
                # Define the weight vector (you can adjust the weights to suit your needs)
                weight_vector = np.array([1, 0.55, 0.75])

                # Calculate the weighted sum of each row's components
                weighted_sums = np.dot(W_norm, weight_vector)

                # Sort the stacked bar plot by the weighted sums
                sort_idx = np.argsort(weighted_sums)[::-1]
                dftoplot = df_mean_imp.iloc[sort_idx, :]
                W_norm = W_norm[sort_idx, :]
                H = H[:, sort_idx]


            else:

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
            ax.set_xticks(ind)

            for i, tick_label in enumerate(ax.get_xticklabels()):
                # Get the country code from the x-axis label
                country_code = tick_label.get_text()
                if country_code in dict_countrycode2info:
                    language_family = dict_countrycode2info[country_code][3]
                    if language_family in dict_langfam2color:
                        tick_label.set_color(dict_langfam2color[language_family])
                # Set the font weight of the tick label to bold
                tick_label.set_fontweight('bold')
                # Set the font size of the tick label to 9
                tick_label.set_fontsize(9)
                # Set the font name of the tick label to the specified font
                tick_label.set_fontname(font_name)

            # Add a legend and title to the plot
            ax.legend()
            ax.set_title(f"Non-Negative Matrix Factorization (n_components={n_components})", fontname=font_name)
            # Set the font size of the title to 12
            ax.title.set_fontsize(12)

            plt.yticks(fontname="Times New Roman")
            plt.savefig(prefixoutput+"_nmf_c"+str(n_components)+"_imp.png", dpi=1200)
            plt.show()
            if n_components == 3:
                # Call the plot_triangle function with W_norm, labels, and colors
                labels = dftoplot.index.tolist()
                colors = [dict_langfam2color[dict_countrycode2info[country_code][3]] if country_code in dict_countrycode2info and dict_countrycode2info[country_code][3] in dict_langfam2color else 'k' for country_code in labels]

                plot_triangle(W_norm, labels, colors, title=f"Non-Negative Matrix Factorization (n_components={n_components})\nTriangle Plot (barycentric coordinate)", filename=prefixoutput+"_nmf_c"+str(n_components)+"_triangle.png")














        ################################################################

        #UMAP

        # Prepare the data
        X = df_mean_imp.iloc[:, :-1]

        # Standardize data
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # Run UMAP
        umap_reducer = umap.UMAP(n_components=2)
        X_umap = umap_reducer.fit_transform(X_std)

        # Create a figure
        fig = plt.figure(figsize=(10, 8))

        # Plot scatter with country colors
        colors = [dict_country2color.get(country, (0, 0, 0)) for country in df_mean_imp.index]
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c=colors)

        # Plot labels for data points
        plt.xlabel('UMAP 1', fontproperties=fm.FontProperties(fname=font_path, size=12))
        plt.ylabel('UMAP 2', fontproperties=fm.FontProperties(fname=font_path, size=12))
        for i, label in enumerate(df_mean_imp.index):
            x = X_umap[i, 0]
            y = X_umap[i, 1]
            plt.annotate(label, (x, y), fontproperties=fm.FontProperties(fname=font_path, size=10))

        # Add legends for country color codes
        handles = []
        labels = []
        for lang_fam, color in dict_langfam2color.items():
            handle = plt.scatter([], [], c=[color], label=lang_fam)
            handles.append(handle)
            labels.append(lang_fam)
        plt.legend(handles, labels, fontsize='small')

        # Save and show the figure
        plt.savefig(prefixoutput+"_umap_1_2_imp.png", dpi=1200)
        plt.show()







        #################################################################
        # PAIRWISE DIST
        #################################################################

        #################################################################


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

        def pairwise_variance(row_pvar, df_pvar):
            variances = np.sum(df_pvar + row_pvar, axis=1)
            return variances

        # compute pairwise variance matrix
        var_matrix = df_variance_imp.iloc[:, :-1].apply(lambda row: pairwise_variance(row, df_variance_imp.iloc[:, :-1]), axis=1)

        def pairwise_standard_error(var_matrix, n):
            return np.sqrt(var_matrix) / np.sqrt(n)

        # number of observations, assuming equal sample sizes for all countries
        n = len(df)
        se_matrix = pairwise_standard_error(var_matrix, n)




        def pairwise_ci(dist_matrix, se_matrix, alpha=0.95):
            t_multiplier = stats.t.ppf((1 + alpha) / 2, n - 1)
            ci_lower = dist_matrix - t_multiplier * se_matrix
            ci_upper = dist_matrix + t_multiplier * se_matrix
            return ci_lower, ci_upper

        ci_lower, ci_upper = pairwise_ci(dist_matrix, se_matrix)

        #GR remove:
        #corr_factor, p_value = process_country("USA", dist_matrix, dict_countrycode2info, dict_langfam2color, ci_lower, ci_upper, prefixoutput)
        #print("corr_factor")
        #print(corr_factor, p_value)
        #return df_mean_imp, df_variance, dist_matrix

         # create clustered heatmap using seaborn
        g = sns.clustermap(dist_matrix, cmap='viridis_r', figsize=(12, 12),xticklabels=1,yticklabels=1)
        for i, tick_label in enumerate(g.ax_heatmap.get_ymajorticklabels()):
            tick_label.set_fontproperties(fm.FontProperties(fname=font_path, size=10))
            country_code = tick_label.get_text()
            if country_code in dict_countrycode2info:
                language_family = dict_countrycode2info[country_code][3]
                if language_family in dict_langfam2color:
                    tick_label.set_color(dict_langfam2color[language_family])
        for i, tick_label in enumerate(g.ax_heatmap.get_xmajorticklabels()):
            tick_label.set_fontproperties(fm.FontProperties(fname=font_path, size=10))
            country_code = tick_label.get_text()
            if country_code in dict_countrycode2info:
                language_family = dict_countrycode2info[country_code][3]
                if language_family in dict_langfam2color:
                    tick_label.set_color(dict_langfam2color[language_family])



        #g = sns.clustermap(dist_matrix, cmap='viridis_r', figsize=(12, 12), xticklabels_rotation=90, yticklabels_rotation=0)

        # set row and column labels
        g.ax_heatmap.set_xlabel("Columns")
        g.ax_heatmap.set_ylabel("Rows")

        #@MB color the country labels
        # display the heatmap
        plt.savefig(prefixoutput+"_nmf_pairwise_dendrogram_imp.png", dpi=1200)
        plt.show()

        # Get the list of countries in the distance matrix
        countries = dist_matrix.index
        results = {}

        for country in countries:
            print("processing "+country)
            ##return corr_factor, p_value
            #out = process_country(country, dist_matrix, dict_countrycode2info, dict_langfam2color, ci_lower, ci_upper, prefixoutput)
            #print(type(out), out if not hasattr(out, "__len__") else len(out))

            result1, result2 = process_country(country, dist_matrix, dict_countrycode2info, dict_langfam2color, ci_lower, ci_upper, prefixoutput)
            results[country] = (result1, result2)



        # Sort the dictionary by values
        sorted_results = dict(sorted(results.items(), key=lambda item: item[1][0]))

        # Get country names and corresponding result1 values
        countries = list(sorted_results.keys())
        result1_values = [result[0] for result in sorted_results.values()]

        plt.figure(figsize=(10,6))
        plt.bar(countries, result1_values)
        plt.xlabel('Country')
        plt.ylabel('Result1')
        plt.title('Correlation with English proficiency per country')
        plt.xticks(rotation=90)  # Rotate x-tick labels for readability
        plt.savefig(prefixoutput+"_eng_pro_corr.png", dpi=1200)
        plt.show()


# Now results is a dictionary with country codes as keys and tuples of results as values
#         with ProcessPoolExecutor(max_workers=35) as executor:
#             # Store the futures in a list
#             futures = [executor.submit(process_country, country, dist_matrix, dict_countrycode2info, dict_langfam2color, ci_lower, ci_upper, prefixoutput) for country in countries]

#             # Iterate over the futures as they complete
#             for future in as_completed(futures):
#                 result = future.result()
#                 print(result)


#         print("Starting execution")
#         with ProcessPoolExecutor(max_workers=35) as executor:
#             print("Created executor")
#             # Store the futures in a list
#             futures = [executor.submit(process_country, country, dist_matrix, dict_countrycode2info, dict_langfam2color, ci_lower, ci_upper, prefixoutput) for country in countries]
#             print("Submitted jobs")
#             try:
#                 # Iterate over the futures as they complete
#                 for future in as_completed(futures):
#                     print("Job completed")
#                     result = future.result()
#                     print("Got result:", result)
#             except Exception as e:
#                 print(f"An error occurred: {e}")

#         print("Execution finished")

#         results_corr = {}
#         for country, future in zip(countries, as_completed(futures)):
#             result = future.result()
#             results_corr[country] = result  # Store the tuple (value1, value2) under the key 'country'

        return df_mean_imp, df_variance, dist_matrix

    else:
        ## Use multiple cores to process the countries
        #with ProcessPoolExecutor(max_workers=35) as executor:
        #    executor.map(process_country, countries)
        #create_bar_plot('USA',  dist_matrix, dict_countrycode2info, dict_langfam2color, ci_lower, ci_upper, prefixoutput)
        return df_mean_imp, df_variance, dist_matrix


# In[85]:




# In[24]:


df_filter_all=df.copy()
df_filter_all=df_filter_all.loc[(df_filter_all['Q263'].isin([1, -4])) & (df_filter_all['Q264'].isin([1, -4])) & (df_filter_all['Q265'].isin([1, -4]))]

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
    conditions.append((df_filter_all['COUNTRIES'] == country) & (df_filter_all['Q272'].isin(q272_values)))

condition = np.logical_or.reduce([c.to_numpy() for c in conditions])
condition = pd.Series(condition, index=df_filter_all.index)

df_filter_all = df_filter_all.loc[condition | (~df_filter_all['COUNTRIES'].isin(countries_q272.keys()))]


# conditions = []
# for country, q272_values in countries_q272.items():
#     conditions.append((df_filter_all['COUNTRIES'] == country) & (df_filter_all['Q272'].isin(q272_values)))

# condition = pd.concat(conditions, axis=0).any(level=0)

# df_filter_all = df_filter_all.loc[condition | (~df_filter_all['COUNTRIES'].isin(countries_q272.keys()))]


# In[25]:


# In[83]:

#Unfiltered dataframe (with immigrants)
df_means, df_var, df_pwdist=performAnalyses(df,output_dir+"unfiltered", 1)
#print(df_pwdist.columns.tolist())
# In[57]:
#Filtered dataframe based on immigration and language spoken at home
df_means_filtered, df_var_filtered, df_pwdist_filtered=performAnalyses(df_filter_all,output_dir+"filtered", 0)
print(df_pwdist_filtered.columns.tolist())


# In[26]:


#Splitting Canada into French Canada and English Canada and splitting Kazakhstan into Russian KZ and Kazakh KZ
df_CANKZ=df_filter_all.copy()
df_CANKZ.loc[(df_CANKZ['COUNTRIES'] == 'CAN') & (df_CANKZ['Q272'] == 1240), 'COUNTRIES'] = 'CDE'
df_CANKZ.loc[(df_CANKZ['COUNTRIES'] == 'CAN') & (df_CANKZ['Q272'] == 1400), 'COUNTRIES'] = 'CDF'
df_CANKZ.loc[(df_CANKZ['COUNTRIES'] == 'KAZ') & (df_CANKZ['Q272'] == 2230), 'COUNTRIES'] = 'KZK'
df_CANKZ.loc[(df_CANKZ['COUNTRIES'] == 'KAZ') & (df_CANKZ['Q272'] == 3630), 'COUNTRIES'] = 'KZR'


# In[27]:


("#Filtered", "dataframe", "with", "a", "split", "for", "countries", "with", "more", "than", "one", "official", "language(CAN", "and", "KAZ", "in", "our", "case)")
df_means_CANKZ_filter, df_var_CANKZ_filter, df_pwdist_CANKZ_filter=performAnalyses(df_CANKZ,output_dir+"CANKAZ_filter",0)
print(df_pwdist_CANKZ_filter.columns.tolist())


#In[81]:


# In[28]:


# #Splitting Canada into French Canada and English Canada and splitting USA into Republicans and Democrats
# df_USpolitics=df_filter_all.copy()
# df_USpolitics = df_USpolitics.loc[~((df_USpolitics['COUNTRIES'] == 'USA') & (~df_USpolitics['Q223'].isin(['USR', 'USD'])))]
# df_USpolitics.loc[(df_USpolitics['COUNTRIES'] == 'CAN') & (df_USpolitics['Q272'] == 1240), 'COUNTRIES'] = 'CDE'
# df_USpolitics.loc[(df_USpolitics['COUNTRIES'] == 'CAN') & (df_USpolitics['Q272'] == 1400), 'COUNTRIES'] = 'CDF'
# df_USpolitics.loc[(df_USpolitics['COUNTRIES'] == 'USA') & (df_USpolitics['Q223'] ==840001 ), 'COUNTRIES'] = 'USR'
# df_USpolitics.loc[(df_USpolitics['COUNTRIES'] == 'USA') & (df_USpolitics['Q223'] == 840002), 'COUNTRIES'] = 'USD'
# print(df_USpolitics.head())


# --- Copy original filtered dataframe ---
df_USpolitics = df_filter_all.copy()

# --- Ensure numeric codes ---
df_USpolitics["Q223_num"] = pd.to_numeric(df_USpolitics["Q223"], errors="coerce")
df_USpolitics["Q272_num"] = pd.to_numeric(df_USpolitics["Q272"], errors="coerce")

# --- Split USA into Republicans / Democrats ---
df_USpolitics.loc[
    (df_USpolitics["COUNTRIES"] == "USA") &
    (df_USpolitics["Q223_num"] == 840001),
    "COUNTRIES"
] = "USR"

df_USpolitics.loc[
    (df_USpolitics["COUNTRIES"] == "USA") &
    (df_USpolitics["Q223_num"] == 840002),
    "COUNTRIES"
] = "USD"

# --- REMOVE remaining USA respondents (independents, missing, etc.) ---
df_USpolitics = df_USpolitics.loc[df_USpolitics["COUNTRIES"] != "USA"]

# --- Split Canada into English / French ---
df_USpolitics.loc[
    (df_USpolitics["COUNTRIES"] == "CAN") &
    (df_USpolitics["Q272_num"] == 1240),
    "COUNTRIES"
] = "CDE"   # English Canada

df_USpolitics.loc[
    (df_USpolitics["COUNTRIES"] == "CAN") &
    (df_USpolitics["Q272_num"] == 1400),
    "COUNTRIES"
] = "CDF"   # French Canada

# --- REMOVE remaining CAN respondents (other languages / missing) ---
df_USpolitics = df_USpolitics.loc[df_USpolitics["COUNTRIES"] != "CAN"]

# --- Optional: drop helper columns ---
df_USpolitics = df_USpolitics.drop(columns=["Q223_num", "Q272_num"])

# --- Sanity check ---
#print(df_USpolitics["COUNTRIES"].value_counts())


# In[29]:


# In[ ]:


#Filtered dataframe where USA is split based on political parties voters and CAN into French and English
df_means_USPOL_filter, df_var_USPOL_filter, df_pwdist_USPOL_filter=performAnalyses(df_USpolitics,output_dir+"USAPolitics_filter", 0)
print(df_pwdist_USPOL_filter.columns.tolist())


# In[30]:


#Filtered dataframe where USA is split into North and South based on latitude and Canada into French and English
#@MB was this: I changed to filter:
#df_USLT=df
df_USLT=df_filter_all.copy()
df_USLT.loc[(df_USLT['COUNTRIES'] == 'CAN') & (df_USLT['Q272'] == 1240), 'COUNTRIES'] = 'CDE'
df_USLT.loc[(df_USLT['COUNTRIES'] == 'CAN') & (df_USLT['Q272'] == 1400), 'COUNTRIES'] = 'CDF'
df_USLT.loc[(df_USLT['COUNTRIES'] == 'USA') & (df_USLT['O2_LATITUDE'] >= 39 ), 'COUNTRIES'] = 'USS'
df_USLT.loc[(df_USLT['COUNTRIES'] == 'USA') & (df_USLT['O2_LATITUDE'] < 39), 'COUNTRIES'] = 'USN'

print(df_USLT.head())
#df_USLT_filter, df_var_USLT_filter, df_pwdist_USLT_filter=performAnalyses(df_USLT,output_dir+"USALat_filter", 0)
#print(df_pwdist_USLT_filter.columns.tolist())


# In[31]:


#Filtering dataframe based on second most spoken language in high immigration countries:
df_SecLang=df.copy()
df_SecLang.loc[(df_SecLang['COUNTRIES'] == 'CAN') & (df_SecLang['Q272'] == 1240), 'COUNTRIES'] = 'CANCHN' #Cantonese
df_SecLang.loc[(df_SecLang['COUNTRIES'] == 'USA') & (df_SecLang['Q272'] == 1270), 'COUNTRIES'] = 'USASPN' #Spanish; Castilian
df_SecLang.loc[(df_SecLang['COUNTRIES'] == 'GBR') & (df_SecLang['Q272'] == 3520), 'COUNTRIES'] = 'GRBPLS' #Polish
df_SecLang.loc[(df_SecLang['COUNTRIES'] == 'AUS') & (df_SecLang['Q272'] == 2870), 'COUNTRIES'] = 'AUSCHN' #Standard Chinese; Mandarin; Putonghua; Guoy
df_SecLang.loc[(df_SecLang['COUNTRIES'] == 'NZL') & (df_SecLang['Q272'] == 2870), 'COUNTRIES'] = 'NZLCHN' #Standard Chinese; Mandarin; Putonghua; Guoy
df_SecLang.loc[(df_SecLang['COUNTRIES'] == 'DEU') & (df_SecLang['Q272'] == 4370), 'COUNTRIES'] = 'DEUTRK' #Turkish
df_SecLang.loc[(df_SecLang['COUNTRIES'] == 'NLD') & (df_SecLang['Q272'] == 1240), 'COUNTRIES'] = 'NLDENG' #English
#Filtered dataframe based on second most spoken language
df_SecLang_filter, df_var_SecLang_filter, df_pwdist_SecLang_filter=performAnalyses(df_SecLang,output_dir+"SecondLanguageFilter", 0)
print(df_pwdist_SecLang_filter.columns.tolist())



# In[32]:


#Filtered dataframe for age range 16-29
#@MB was this: I changed to filter:
df_y=df_filter_all.copy()
df_y=df_y[df_y['X003R2'] == 1]
#print(df_y.head(10))
#print(df_y.shape)
#print(df_y['Q82_EU'])
#print(df_y.columns)
df_means_y_filter, df_var_y_filter, df_pwdist_y_filter=performAnalyses(df_y,output_dir+"16-29_Filter", 0)
print(df_pwdist_y_filter.columns.tolist())


# In[ ]:


#Filtered dataframe for age range 30-49
#@MB was this: I changed to filter:
#CHANGE THIS TO COPY
df_m=df_filter_all.copy()

df_m=df_m[df_m['X003R2'] == 2]

#print(df_m.head(10))
#print(df_m.shape)

df_means_m_filter, df_var_m_filter, df_pwdist_m_filter=performAnalyses(df_m,output_dir+"30-49_Filter",0)
print(df_pwdist_m_filter.head())


# In[ ]:


#Filtered dataframe for age over 50
df_o=df_filter_all.copy()
df_o=df_o[df_o['X003R2'] == 3]

print(df_o.head(10))

df_means_o_filter, df_var_o_filter, df_pwdist_o_filter=performAnalyses(df_o,output_dir+"50andOver_Filter", 0)


# In[ ]:


def compare_dataframes(df1, df2):
    # Check if the dataframes have the same shape
    if df1.shape != df2.shape:
        print("The dataframes have different shapes!")
        return

    # Check if the dataframes have the same columns
    if not df1.columns.equals(df2.columns):
        print("The dataframes have different columns!")
        return

    # Create a boolean mask for where the dataframes are not equal
    diff_mask = df1 != df2

    return diff_mask


def write_questions_to_file(df, dict_questions, txt_file, latex_file):
    df = df.to_frame()  # Convert Series to DataFrame

    with open(txt_file, 'w') as txt, open(latex_file, 'w') as latex:
        latex.write("\\begin{tabular}{|l|l|l|}\n")
        latex.write("\\hline\n")
        latex.write("Country & Question & Category \\\\ \\hline\n")
        for country, row in df.iterrows():
            questions = row[0]  # Access the first (and only) column
            for question_code in questions:
                question = dict_questions.get(question_code, {"question": "Unknown", "category": "Unknown"})
                txt.write(f"Country: {country}, Question: {question['question']}, Category: {question['category']}\n")
                latex.write(f"{country} & {question['question']} & {question['category']} \\\\ \\hline\n")
        latex.write("\\end{tabular}")


def compare2dataframes(df1, df2, outputprefix, title):
    compare_dataframes(df1, df2)
    df1_cpy = df1.copy()

    df1_cpy.drop('language_family', axis=1, inplace=True)

    df2_cpy = df2.copy()
    df2_cpy.drop('language_family', axis=1, inplace=True)

    # Calculate absolute difference between the two dataframes
    diff = (df1_cpy - df2_cpy).abs()

    # Find the columns with the 5 largest and 5 smallest differences for each row
    largest_diff = diff.apply(lambda row: row.nlargest(5).index.tolist(), axis=1)
    smallest_diff = diff.apply(lambda row: row.nsmallest(5).index.tolist(), axis=1)
    print(largest_diff)
    write_questions_to_file(largest_diff, dict_questions, str(outputprefix)+'_top5_largestdiff_questions.txt', str(outputprefix)+'_top5_largestdiff_questions.tex')
    write_questions_to_file(smallest_diff, dict_questions, str(outputprefix)+'_top5_smallestdiff_questions.txt', str(outputprefix)+'_top5_smallestdiff_questions.tex')

    distances = np.linalg.norm(df1_cpy.values - df2_cpy.values, axis=1)

    # Create a DataFrame with the distances
    df_distances = pd.DataFrame(distances, columns=['Distance'], index=df1_cpy.index)

    # Sort the DataFrame
    df_distances_sorted = df_distances.sort_values(by='Distance')

    # Create the bar plot
    ax = df_distances_sorted.plot(kind='bar', legend=False, figsize=(10, 5))

    # Set the title
    ax.set_title("Distance between "+title)

    # Set font properties
    font_path = fm.findfont(fm.FontProperties(family='monospace'))
    font_name = fm.FontProperties(fname=font_path).get_name()

    for i, tick_label in enumerate(ax.get_xticklabels()):
        tick_label.set_fontproperties(fm.FontProperties(fname=font_path, size=8))
        country_code = tick_label.get_text()
        if country_code in dict_countrycode2info:
            language_family = dict_countrycode2info[country_code][3]
            if language_family in dict_langfam2color:
                tick_label.set_color(dict_langfam2color[language_family])
        tick_label.set_rotation(90)  # To make labels horizontal

    # Show the plot
    plt.savefig(outputprefix+"_barplot_diff_orted.png", dpi=1200)
    plt.show()


# In[ ]:


compare2dataframes(df_means_y_filter,df_means_o_filter,output_dir+"yvso_filter","younger vs older")


# In[ ]:


#df_filter_all['Q260'].value_counts()
df_filter_male=df_filter_all.copy()
df_filter_male=df_filter_male[df_filter_male['Q260'] == 1]
df_filter_male_means, df_filter_male_var, df_filter_male_pwdist=performAnalyses(df_filter_male,output_dir+"male", 0)


# In[ ]:



#df_filter_all['Q260'].value_counts()
df_filter_female=df_filter_all.copy()
df_filter_female=df_filter_female[df_filter_female['Q260'] == 2]
df_filter_female_means, df_filter_female_var, df_filter_female_pwdist=performAnalyses(df_filter_female,output_dir+"female", 0)


# In[ ]:


compare2dataframes(df_filter_male_means,df_filter_female_means,output_dir+"malevsfemale_filter","males vs females")


# In[ ]:


#Filtered dataframe for age range 16-29

dfCADUSA_y=df_USpolitics.copy()
dfCADUSA_y=dfCADUSA_y[dfCADUSA_y['X003R2'] == 1]

dfCADUSA_means_y_filter, dfCADUSA_var_y_filter, dfCADUSA_pwdist_y_filter=performAnalyses(dfCADUSA_y,output_dir+"16-29_CADUSA", 0)
#print(df_pwdist_y_filter.columns.tolist())


# In[ ]:


dfCADUSA_m=df_USpolitics.copy()

dfCADUSA_m=dfCADUSA_m[dfCADUSA_m['X003R2'] == 2]

#print(df_m.head(10))
#print(df_m.shape)

dfCADUSA_means_m_filter, dfCADUSA_var_m_filter, dfCADUSA_pwdist_m_filter=performAnalyses(dfCADUSA_m,output_dir+"30-49_CADUSA",0)


# In[ ]:


dfCADUSA_o=df_USpolitics.copy()
dfCADUSA_o=dfCADUSA_o[dfCADUSA_o['X003R2'] == 3]

#print(dfCADUSA_o.head(10))

dfCADUSA_means_o_filter, dfCADUSA_var_o_filter, dfCADUSA_pwdist_o_filter=performAnalyses(dfCADUSA_o,output_dir+"50andOver_CADUSA", 0)



