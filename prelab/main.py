import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('PreLab_Data1.csv')
print(df.head())

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(df['Time (min)'], df['Progress (%)'], color='blue', label='Progress')
ax1.set_xlabel('Time (mins)')
ax1.set_ylabel('Progress (%)')

ax2 = ax1.twinx()
ax2.plot(df['Time (min)'], df['Frustration (1-10 scale)'], color='red', label='Frustration')
ax2.set_ylabel('Frusteration (1-10)')

plt.title('Progress and Frustration Over Time')
plt.show()