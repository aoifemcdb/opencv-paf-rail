import pandas as pd
import matplotlib.pyplot as plt

sensorname = '053-1478@192.168.0.100'

df_10 = pd.read_csv('./silicone_samples/test.csv')
# df_20 = pd.read_csv('./silicone_samples/ds20.csv')
# df_30 = pd.read_csv('./silicone_samples/ds30.csv')
# df_40 = pd.read_csv('./silicone_samples/ss940.csv')
# df_50 = pd.read_csv('./silicone_samples/ss950.csv')

fig = plt.figure()

# df = df.rename(columns = {'Seconds': 'Time (s)', '053-1478@192.168.0.100': 'Force (g)'})
ax = df_10.plot(x='Time (s)', y = 'Force (g)', label = 'test')
# df_20.plot(y = 'Force (g)', ax=ax, label = 'DS20')
# df_30.plot(y = 'Force (g)', ax=ax, label = 'DS30')
# df_40.plot(y = 'Force (g)', ax=ax, label = 'SS940')
# df_50.plot(y = 'Force (g)', ax=ax, label = 'SS950')
plt.ylabel('Force (g)')
# plt.legend('DS10', 'DS20', 'DS30', 'SS940', 'SS950')
plt.show()

