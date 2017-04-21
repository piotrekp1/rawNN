import seaborn as sns
import matplotlib.pyplot as plt

# df = sns.load_dataset("anscombe")

# Show the results of a linear regression within each dataset
# sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
#           col_wrap=2, ci=None, palette="muted", size=4,
#           scatter_kws={"s": 50, "alpha": 1})
from pandas import DataFrame

d = {'eta': [3, 4], 'ilosc_powtorzen': [3, 4]}
df = DataFrame(data=d)

sns.pointplot(x="eta", y="ilosc_powtorzen", data=df)
plt.show()
