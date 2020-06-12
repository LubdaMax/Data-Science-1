import nltk

# Initialize the dataframe used to store Articles in col. 0
index = np.linspace(0, 509,510)
total_data = pd.DataFrame(columns=["Article"],["Category"], index=index)
total_data = total_data.fillna(0)

# Relative path of dataset
rootpath = Path.cwd()
articlepath_folder = Path.joinpath(rootpath, r"Dataset 1 (BBC)\News Articles\")
summarypath_folder = Path.joinpath(rootpath, r"Dataset 1 (BBC)\Summaries\")



# Insert data (ALL articles) from .txt files into the dataframe
#topics = ["business", "entertainment","politics","sport","tech"]

# Insert data (only business articles) from .txt files into the dataframe
topics = ["business"]

count = 0
for topic in topics:
    articlepath = articlepath_folder + topic
    for entry in os.scandir(articlepath):
        text_file = open(entry, "r")
        raw_text = text_file.read()
        total_data.iloc[count, 0] = raw_text
        total_data.iloc[count,1] = topic
        text_file.close()

        count += 1


