import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime

models = [
    ("Gemini-2.5-Pro", "2025-03-25", 63.3),
    ("Gemini-2.5-Flash", "2025-04-17", 62.1),
    ("Qwen3-235B-A22B-Thinking-2507", "2025-07-25", 60.6),
    ("DeepSeek-R1", "2025-01-20", 58.3),
    ("Qwen3-235B-A22B-Instruct-2507", "2025-07-22", 58.3),
    ("o1-preview", "2024-09-12", 57.7),
    ("DeepSeek-R1-0528", "2025-05-28", 56.7),
    ("MiniMax-Text-01", "2025-01-15", 56.5),
    ("Gemini-2.0-Flash-Thinking", "2025-01-21", 56.0),
    ("Gemini-Exp-1206", "2024-12-06", 52.5),
    ("GPT-4o-1120", "2024-11-20", 51.4),
    ("GPT-4o-0806", "2024-08-06", 51.2),
    ("Gemini-2.0-Flash", "2024-12-11", 51.1),
    ("GLM-4.5", "2025-07-28", 50.3),
    ("Qwen3-30B-A3B-Thinking-2507", "2025-07-22", 50.1),
    ("Qwen3-235B-A22B", "2025-04-29", 50.1),
    ("Qwen3-32B", "2025-04-29", 49.2),
    ("QwQ-32B", "2025-03-06", 48.9),
    ("GLM-4.5-Air", "2025-07-28", 48.6),
    ("Claude 3.5 Sonnet", "2024-10-22", 46.7),
    ("GLM-4-Plus", "2024-10-11", 46.1),
    ("Kimi-K2-Instruct", "2025-07-11", 44.3),
    ("Qwen2.5-72B", "2024-09-19", 43.5),
    ("Qwen3-30B-A3B", "2025-04-29", 42.5),
    ("Mistral Large 24.11", "2024-11-24", 39.6),
    ("o1-mini", "2024-09-12", 38.9),
    ("Llama 3.1 70B", "2024-07-23", 36.2),
    ("Llama 3.3 70B", "2024-12-06", 36.2),
    ("Qwen2.5-7B", "2024-09-19", 35.6),
    ("Nemotron 70B", "2024-10-15", 35.2),
    ("Mistral Large 2", "2024-07-24", 33.6),
    ("GPT-4o mini", "2024-07-18", 32.4)
]

df = pd.DataFrame(models, columns=["Model", "Date", "Score"])
df["Date"] = pd.to_datetime(df["Date"])

import seaborn as sns
palette = sns.color_palette("hls", len(df))

fig, ax = plt.subplots(figsize=(10, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for i, row in df.iterrows():
    ax.scatter(row["Date"], row["Score"],
               color=palette[i], s=70, label=row["Model"], edgecolors='k', linewidths=0.7, zorder=3)
    ax.annotate(row["Model"],
                xy=(row["Date"], row["Score"]),
                xytext=(5,5),  #右方微微1像素上,你可调明显一点如(4,0)
                textcoords='offset points',
                fontsize=10, color=palette[i], alpha=1,
                ha='left', va='center')
    
for ref in [35, 40, 45, 50, 55, 60, 65]:
    ax.axhline(ref, color='lightgray', ls=':', lw=1, zorder=1)

human_y = 53.7
ax.axhline(human_y, ls='--', lw=1.7, color='gray', label='Human baseline')
ax.text(df['Date'].min(), human_y+0.3, "Human Baseline (53.7%)", color='gray', fontsize=10, va='bottom')

ax.set_xlabel('Release Date', fontsize=14)
ax.set_ylabel('Overall w/ CoT (%)', fontsize=14)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("static/images/leaderboard.png", dpi=200)
plt.show()