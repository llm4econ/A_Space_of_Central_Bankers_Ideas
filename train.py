from datetime import datetime
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num, num2date
from matplotlib import cm
import matplotlib.patheffects as pe
import plotly.express as px
from dotenv import load_dotenv

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN


load_dotenv()

def bertopic_modeling(
    docs,
    embeddings,
    n_gram_range=(1, 4),
    min_cluster_size=50,
    top_n_words=10,
    umap_n_neighbors=30,
    umap_n_components=20,
    umap_min_dist=0,
    umap_metric='euclidean',
    umap_random_state=42,
    embedding_model_name="BAAI/bge-base-en-v1.5",
):
    embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=umap_random_state,
    )
    umap_model.fit(embeddings)
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=n_gram_range)
    ctfidf_model = ClassTfidfTransformer()
    representation_model = KeyBERTInspired()
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        top_n_words=top_n_words,
    )
    topics, probabilities = topic_model.fit_transform(docs, embeddings)
    return topic_model, topics, probabilities

def plot_bar_chart(data, x_label, y_label, title, output_dir, filename, kind='bar', horizontal=False):
    plt.figure(figsize=(5, 5))
    if horizontal:
        data.plot(kind=kind, color='skyblue', edgecolor='black')
    else:
        data.plot(kind=kind, color='skyblue', edgecolor='black')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=0, ha='center', fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    pdf_path = f'{output_dir}/{filename}.pdf'
    png_path = f'{output_dir}/{filename}.png'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print("Saved to", pdf_path)

os.makedirs("plots", exist_ok=True)
os.makedirs("bertopic_models", exist_ok=True)

df = pd.read_parquet("hf://datasets/tpark-bis/central_bank_speeches/central_bank_speeches.parquet")
df.drop(columns=["topic_vector"], inplace=True)
output_dir_plots = "plots"
output_dir_models = "bertopic_models"

central_bank_counts = df['country_iso2'].value_counts()
formatted_list = [f"{bank} ({count})" for bank, count in central_bank_counts.items()]
output_text = ", ".join(formatted_list)
print(output_text)

df_affiliation = df['country_iso2'].value_counts().reset_index()
df_affiliation.columns = ['Central bank', 'Count']
df_affiliation['Count'] = df_affiliation['Count'].astype(int)

df_affiliation = df_affiliation.set_index('Central bank')
central_bank_counts = df_affiliation['Count'].sort_values(ascending=False).head(10)
plot_bar_chart(central_bank_counts, "Location of central bank", "# of speeches", "", output_dir_plots, "fig_stat_by_cb", kind='bar')

df['date'] = pd.to_datetime(df['date'])
counts_by_year = df['date'].dt.year.value_counts().sort_index()
counts_by_year.index = counts_by_year.index.map(lambda x: f"{x % 100:02d}")
plot_bar_chart(counts_by_year, "Year", "# of speeches", "", output_dir_plots, "fig_stat_by_year", kind='bar')

top_authors = df['speaker'].value_counts().head(10)
plot_bar_chart(top_authors, "# of speeches", "Speaker", "", output_dir_plots, "fig_stat_by_speaker", kind='barh', horizontal=True)






major_central_banks = ["AU", "CA", "DK", "EU", "JP", "NZ", "NO", "SE", "CH", "GB", "US", "DZ", "AR", "BR", "CL", "CN", "CO", "CZ", "HK", "HU", "IN", "ID", "IL", "KR", "KW", "MY", "MX", "MA", "PE", "PH", "PL", "RO", "RU", "SA", "SG", "ZA", "TH", "TR", "AE", "VN"]
major_bank = df[df['country_iso2'].isin(major_central_banks)].copy()
major_bank.reset_index(drop=True, inplace=True)
docs = major_bank['summary'].tolist()
major_bank_embeddings = major_bank['embeddings'].tolist()
major_bank_embeddings = np.array(major_bank_embeddings)

topic_model, topics, probabilities = bertopic_modeling(
    docs=docs,
    embeddings=major_bank_embeddings,
    n_gram_range=(1, 4),
    min_cluster_size=50,
    top_n_words=10,
    umap_n_neighbors=30,
    umap_n_components=20,
    umap_min_dist=0,
    umap_metric='euclidean',
    umap_random_state=23,
)

df_topic = topic_model.get_topic_info()
reduced_embeddings = UMAP(n_components=2, n_neighbors=30, min_dist=0.6, random_state=42).fit_transform(major_bank_embeddings)
topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings, hide_document_hover=True, hide_annotations=True)
df_reduced_embeddings = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
major_bank['x'] = df_reduced_embeddings['x']
major_bank['y'] = df_reduced_embeddings['y']
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"{output_dir_models}/G1_model_{current_datetime}"
topic_model.save(model_name, serialization="safetensors")
print(model_name)

ctfidf_matrix = topic_model.c_tf_idf_
feature_names = topic_model.vectorizer_model.get_feature_names_out()
ctfidf_dense = ctfidf_matrix.todense()
ctfidf_df = pd.DataFrame(ctfidf_dense, columns=feature_names)
top_30_df = pd.DataFrame(columns=["Keyword_Ranking"])

for i in range(len(topic_model.generate_topic_labels())):
    top_terms = ctfidf_df.loc[i].sort_values(ascending=False).head(30)
    lst_top_terms = top_terms.index.tolist()
    top_30_df = pd.concat([top_30_df, pd.DataFrame({"topic_vector": [lst_top_terms]})], ignore_index=True)
    
df_topic = pd.merge(df_topic, top_30_df, how='left', left_index=True, right_index=True)
df_res = pd.DataFrame({'Topic': topics, 'Probability': probabilities})
df_res = df_res.merge(df_topic, how='left', on='Topic')
major_bank = pd.merge(major_bank, df_res, left_index=True, right_index=True, how='left')

file_path = r'Major_Bank_Topics_20250511_234633.parquet'
df_major_topic = major_bank
#df_major_topic = pd.read_parquet(file_path)
df_major_topic = df_major_topic[['date', 'title', 'summary', 'speaker', 'affiliation', 'x', 'y', 'Name', 'Representation', 'Topic', 'topic_vector', 'Probability', 'country_iso2']]
df_major_topic_list = df_major_topic[['Name', 'topic_vector', 'Topic']]
df_major_topic_list = df_major_topic_list.drop_duplicates(subset=['Topic'])
df_major_topic_list.reset_index(drop=True, inplace=True)
df_major_topic_list = df_major_topic_list.sort_values(by='Topic')


### PLACEHOLDER - Create your own mappings here
NAME_TO_LABEL = {
    "-1_Financial stability policies": "General",
    "0_Euro Area Stability": "ECB",
    "1_Federal Reserve Policy": "Fed",
    "2_Japan Economic Outlook": "JP",
    "3_Central Bank Digital Currencies": "Payment",
    "4_Philippine Financial Stability": "PH",
    "5_Financial regulation reforms": "Fin. Reg.",
    "6_Canadian Monetary Policy": "CA",
    "7_Global Capital Flows": "Global Econ.",
    "8_Australian Economic Outlook": "AU",
    "9_Swedish Monetary Policy": "SE",
    "10_Swiss Monetary Policy": "CH",
    "11_South Africa Economy": "ZA",
    "12_Asia Financial Integration": "Asia",
    "13_UK Monetary Policy": "GB",
    "14_Norwegian Monetary Policy": "NO",
    "15_ECB monetary policy": "ECB",
    "16_EU Banking Integration": "ECB",
    "17_India's Economic Policy": "IN",
    "18_Climate Financial Risks": "Climate",
    "19_Global Islamic Finance": "Islamic",
    "20_New Zealand Economy": "NZ",
    "21_Monetary and Financial Stability": "Stability",
    "22_Hong Kong Finance": "HK",
    "23_Corporate Risk Management": "Governance",
    "24_Thailand Economic Stability": "TH",
    "25_Financial Inclusion": "Fin. Inclusion",
    "26_Monetary Policy Stability": "Price stability",
    "27_CRA Housing Reform": "Housing",
    "28_Indian Banking Reform": "IN",
    "29_ECB monetary policy": "ECB",
    "30_Chile Economic Outlook": "CL",
    "31_Israel Economic Policy": "IL",
    "32_ECB monetary policy": "ECB",
    "33_Denmark fixed exchange": "DK",
    "34_Central Bank Communication": "Comms.",
    "35_Diversity in Economics": "Diversity",
    "36_Turkey Inflation Policy": "TR",
    "37_FX Global Code": "Money Mkt",
    "38_Malaysia Financial Development": "MY",
    "39_Basel II Implementation": "Basel II",
    "40_Central Bank Liquidity": "Liquidity",
    "41_Insurance Industry Development": "Insurance",
    "42_Anti-Money Laundering": "AML",
    "43_Shanghai Financial Center": "CN",
    "44_Enhancing Statistical Data": "Statistics",
    "45_Insurance regulation": "Solvency II",
    "46_Bank of Korea": "KR",
    "47_ECB Crisis Response": "ECB",
    "48_Community Bank Regulation": "Community bank"
}

#df_major_topic_list['Label'] = df_major_topic_list['Name'].map(name_to_label)
#df_major_topic_list['Label'] = df_major_topic_list['Topic'].astype(str) + '_' + df_major_topic_list['Label']
df_major_topic_list['Label'] = df_major_topic_list['Name']

df_major_topic = pd.merge(df_major_topic, df_major_topic_list[['Name', 'Label']], how='left', on='Name')
unique_names = sorted(df_major_topic['Label'].unique())
base_colors = list(cm.tab20.colors) + list(cm.tab20b.colors) + list(cm.tab20c.colors)
n_topics = len([n for n in unique_names if not n.startswith("-1_")])
if n_topics > len(base_colors):
    base_colors = base_colors * ((n_topics // len(base_colors)) + 1)
rng = np.random.default_rng(23)
rng.shuffle(base_colors)
color_iter = iter(base_colors)

color_map = {}
for name in unique_names:
    if name.startswith("-1_"):
        color_map[name] = "#bfbfbf"
    else:
        color_map[name] = next(color_iter)

x_marker_prefixes = ["3_", "5_", "7_", "18_", "21_", "23_", "25_", "26_", "27_", "34_", "35_", "37_", "39_", "40_", "41_", "42_", "44_", "45_", "48_"]
x_marker_list = [
    group['Label'].iloc[0]
    for prefix in x_marker_prefixes
    for _, group in df_major_topic.groupby('Name')
    if _.startswith(prefix)
]
x_marker_set = set(x_marker_list)

def num_prefix(name):
    m = re.match(r'\s*([\-]?\d+)', name)
    return int(m.group(1)) if m else None

fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

for name, group in df_major_topic.groupby('Label'):
    if name.startswith("-1_"):
        ax.scatter(group['x'], group['y'], s=8, marker='o', color=color_map[name], alpha=0.1, linewidths=0)

for name, group in df_major_topic.groupby('Label'):
    if name.startswith("-1_"):
        continue
    marker = 'X' if name in x_marker_set else 'o'
    ax.scatter(group['x'], group['y'], s=8, marker=marker, color=color_map[name], alpha=0.7, linewidths=0, label=name)

for name, group in df_major_topic.groupby('Label'):
    if name.startswith("-1_"):
        continue
    ax.text(group['x'].median(), group['y'].median(), str(num_prefix(name)), fontsize=7, weight='bold', ha='center', va='center', color=color_map[name], path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

handles, labels = ax.get_legend_handles_labels()
pairs = [(h, l) for h, l in zip(handles, labels)]
pairs_sorted = sorted(pairs, key=lambda x: num_prefix(x[1]))
handles_sorted, labels_sorted = zip(*pairs_sorted)

ax.legend(handles_sorted, labels_sorted, bbox_to_anchor=(0.5, 0.05), loc='upper center', ncol=7, frameon=False, fontsize=6, title='Expert label', title_fontsize=7, columnspacing=0.4, handletextpad=0.3)

plt.subplots_adjust(bottom=0.28)

pdf_path = f'{output_dir_plots}/fig_global_topic.pdf'
png_path = f'{output_dir_plots}/fig_global_topic.png'
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
print("Saved to", pdf_path)

prefixes = ('18_', '45_', '25_', '5_', '21_', '7_')
filtered_df = df_major_topic[df_major_topic['Label'].str.startswith(prefixes)]

n = 3
top_affiliations = (
    filtered_df.groupby(['Label', 'country_iso2'])
    .size()
    .groupby(level=0, group_keys=False)
    .nlargest(n)
    .reset_index(name='count')
)

merged_df = filtered_df.merge(
    top_affiliations[['Label', 'country_iso2']],
    on=['Label', 'country_iso2'],
    how='left',
    indicator=True
)
merged_df['country_iso2_cleaned'] = merged_df.apply(
    lambda row: row['country_iso2'] if row['_merge'] == 'both' else 'Other',
    axis=1
)

aff_shares = (
    merged_df.groupby(['Label', 'country_iso2_cleaned'])
    .size()
    .groupby(level=0, group_keys=False)
    .apply(lambda x: x / x.sum())
    .reset_index()
    .rename(columns={0: 'share'})
)

pivot_df = aff_shares.pivot(index='Label', columns='country_iso2_cleaned', values='share').fillna(0)

affiliations = pivot_df.columns

manual_order = [
    "7_Global Econ.",
    "21_Stability",
    "5_Fin. Reg.",
    "25_Fin. Inclusion",
    "45_Solvency II",
    "18_Climate"
]

pivot_df = pivot_df.reindex(manual_order)

custom_colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#bcbd22",
    "#e377c2",
    "#17becf",
    "#8c564b",
]
color_map = {aff: custom_colors[i % len(custom_colors)] for i, aff in enumerate(affiliations)}
color_map['Other'] = 'grey'

fig, ax = plt.subplots(figsize=(9, 9))
y = np.arange(len(pivot_df))
bar_height = 0.9

for i, (topic, row) in enumerate(pivot_df.iterrows()):
    parts = row.drop('Other').sort_values(ascending=False) if 'Other' in row else row.sort_values(ascending=False)
    if 'Other' in row:
        parts['Other'] = row['Other']

    left = 0
    for aff, value in parts.items():
        ax.barh(y[i], value, left=left, height=bar_height, color=color_map[aff])
        if value > 0.05:
            text_x = left + value / 2
            ax.text(text_x, y[i], aff, ha='center', va='center', fontsize=15, color='white', fontweight='bold')
        left += value

short_labels = pivot_df.index
ax.set_yticks(y)
ax.set_yticklabels(short_labels, fontsize=15, rotation=0)
ax.set_xlabel("Share of Affiliation")
ax.set_ylabel('')
plt.tight_layout(rect=[0, 0.05, 0.95, 1])
plt.subplots_adjust(bottom=0.28)

pdf_path = f'{output_dir_plots}/topic_by_cb.pdf'
png_path = f'{output_dir_plots}/topic_by_cb.png'
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

print("Saved to", pdf_path)

df_filtered = df_major_topic[df_major_topic['Label'].str.startswith(('18_', '3_', '26_', '44_', '39_'))]

plot_data = df_filtered.copy()
plot_data['date'] = pd.to_datetime(plot_data['date'])

unique_names = plot_data['Label'].unique()
color_map = cm.get_cmap('Paired', len(unique_names))
name_to_color = {name: color_map(i) for i, name in enumerate(unique_names)}

custom_colors = [
    "#1f77b4",
    "#ff7f0e",
    "#bcbd22",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#4daf4a",
    "#d62728",
    "#17becf"
]


name_to_color = {name: custom_colors[i % len(custom_colors)] for i, name in enumerate(unique_names)}

plot_data['date_num'] = date2num(plot_data['date'])

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')

for name in unique_names:
    subset = plot_data[plot_data['Label'] == name]
    ax.plot(
        subset['x'], subset['date_num'], subset['y'],
        linestyle='', marker='o', markersize=4,
        color=name_to_color[name], alpha=0.7
    )

for name in unique_names:
    subset = plot_data[plot_data['Label'] == name]
    mean_x = subset['x'].mean()
    mean_y = subset['date_num'].mean() * 0.9
    mean_z = subset['y'].mean()
    ax.text(mean_x, mean_y, mean_z, name, fontsize=15, weight='bold')

ax.set_xlabel('')
ax.set_ylabel('Year')
ax.set_zlabel('')
ax.view_init(elev=20, azim=150)
ax.invert_yaxis()
ax.set_xticklabels([])
ax.set_zticklabels([])

date_labels = [num2date(d).strftime('%Y') for d in ax.get_yticks()]
ax.set_yticklabels(date_labels)

ax.grid(True)

plt.tight_layout()

pdf_path = f'{output_dir_plots}/3d_topic.pdf'
png_path = f'{output_dir_plots}/3d_topic.png'
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

print("Saved to", pdf_path)

ecb = major_bank[(major_bank['affiliation'] == 'European Central Bank')]
ecb = ecb.reset_index(drop=True)
ecb = ecb.iloc[:, :12]

ecb_docs = ecb['summary'].tolist()
ecb_embeddings = ecb['embeddings'].tolist()
ecb_embeddings = np.array(ecb_embeddings)

topic_model, topics, probabilities = bertopic_modeling(
    docs=ecb_docs,
    embeddings=ecb_embeddings,
    n_gram_range=(1, 2),
    min_cluster_size=30,
    umap_n_neighbors=3,
    umap_n_components=20,
    umap_min_dist=0,
    umap_metric='euclidean',
    umap_random_state=42,
)

df_topic_ecb = topic_model.get_topic_info()
reduced_embeddings = UMAP(n_components=2, n_neighbors=20, min_dist=0.5, random_state=42).fit_transform(ecb_embeddings)
topic_model.visualize_documents(ecb_docs, reduced_embeddings=reduced_embeddings, hide_document_hover=True, hide_annotations=True)
df_reduced_embeddings_ecb = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
ecb['x'] = df_reduced_embeddings_ecb['x']
ecb['y'] = df_reduced_embeddings_ecb['y']

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"{output_dir_models}/G2A_model_{current_datetime}"
topic_model.save(model_name, serialization="safetensors")
print(model_name)

ctfidf_matrix = topic_model.c_tf_idf_
feature_names = topic_model.vectorizer_model.get_feature_names_out()
ctfidf_dense = ctfidf_matrix.todense()
ctfidf_df = pd.DataFrame(ctfidf_dense, columns=feature_names)

top_30_df = pd.DataFrame(columns=["Keyword_Ranking"])
for i in range(len(topic_model.generate_topic_labels())):
    top_terms = ctfidf_df.loc[i].sort_values(ascending=False).head(30)
    lst_top_terms = top_terms.index.tolist()
    top_30_df = pd.concat([top_30_df, pd.DataFrame({"topic_vector": [lst_top_terms]})], ignore_index=True)

df_topic_ecb = pd.merge(df_topic_ecb, top_30_df, how='left', left_index=True, right_index=True)

df_res = pd.DataFrame({'Topic': topics, 'Probability': probabilities})
df_res = df_res.merge(df_topic_ecb, how='left', on='Topic')
ecb = pd.merge(ecb, df_res, left_index=True, right_index=True, how='left')

df_ecb_topic_list = df_topic_ecb[['Name', 'topic_vector', 'Topic']]
df_ecb_topic_list = df_ecb_topic_list.drop_duplicates(subset=['Topic'])
df_ecb_topic_list.reset_index(drop=True, inplace=True)
df_ecb_topic_list = df_ecb_topic_list.sort_values(by='Topic')


### PLACEHOLDER - Create your own mappings here
TOPIC_TO_LABEL = {
    -1: "General",
     0: "Monetary policy",
     1: "Financial stability",
     2: "Monetary policy",
     3: "Eurozone integration",
     4: "Monetary policy",
     5: "Price stability",
     6: "Banking supervision",
     7: "Pandemic response",
     8: "Economic outlook",
     9: "Financial integration",
    10: "Euro adoption",
    11: "Climate change",
    12: "Governance",
    13: "Monetary policy",
    14: "Monetary policy",
    15: "Study of MP Impact on Households",
    16: "Payment",
    17: "Structural reform",
    18: "Liquidity",
    19: "Integration challenges"
}
def plot_ecb_topics(ecb, topic_to_label, output_dir, filename):
    df_ecb_topic_list = ecb[['Name', 'Topic']].drop_duplicates()
    # df_ecb_topic_list['Label'] = df_ecb_topic_list['Topic'].map(topic_to_label)
    # df_ecb_topic_list['Label'] = df_ecb_topic_list['Topic'].astype(str) + '_' + df_ecb_topic_list['Label']
    df_ecb_topic_list["Label"] = df_ecb_topic_list["Name"]
    ecb = pd.merge(ecb, df_ecb_topic_list[['Name', 'Label']], how='left', on='Name')

    unique_names = sorted(ecb['Label'].unique())
    base_colors = list(cm.tab20.colors) + list(cm.tab20b.colors) + list(cm.tab20c.colors)
    color_iter = iter(base_colors)
    color_map = {}
    for name in unique_names:
        if name.startswith("-1_"):
            color_map[name] = "#bfbfbf"
        else:
            try:
                color_map[name] = next(color_iter)
            except StopIteration:
                color_map[name] = 'grey'

    x_marker_list = []
    x_marker_set = set(x_marker_list)

    def num_prefix(name):
        m = re.match(r'\s*([\-]?\d+)', name)
        return int(m.group(1)) if m else None

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    for name, group in ecb.groupby('Label'):
        if name.startswith("-1_"):
            ax.scatter(group['x'], group['y'], s=8, marker='o', color=color_map[name], alpha=0.2, linewidths=0)

    for name, group in ecb.groupby('Label'):
        if name.startswith("-1_"):
            continue
        marker = 'X' if name in x_marker_set else 'o'
        ax.scatter(group['x'], group['y'], s=8, marker=marker, color=color_map[name], alpha=0.8, linewidths=0, label=name)

    for name, group in ecb.groupby('Label'):
        if name.startswith("-1_"):
            continue
        ax.text(group['x'].median(), group['y'].median(), str(num_prefix(name)), fontsize=9, weight='bold', ha='center', va='center', color=color_map[name], path_effects=[pe.withStroke(linewidth=1, foreground="black")])

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    pairs = [(h, l) for h, l in zip(handles, labels)]
    pairs_sorted = sorted(pairs, key=lambda x: num_prefix(x[1]))
    handles_sorted, labels_sorted = zip(*pairs_sorted)

    ax.legend(handles_sorted, labels_sorted, bbox_to_anchor=(1.05, 0.5), loc='center left', ncol=1, frameon=False, fontsize=7, title='Indicative manual label', title_fontsize=7, columnspacing=0.4, handletextpad=0.3)

    plt.subplots_adjust(bottom=0.28)
    pdf_path = f'{output_dir}/{filename}.pdf'
    png_path = f'{output_dir}/{filename}.png'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Saved to {pdf_path}")

    return ecb

ecb = plot_ecb_topics(ecb, TOPIC_TO_LABEL, output_dir="plots", filename="fig_ecb_topics")
ecb_mp = ecb[
    (ecb["Name"].str.contains(r"\b(0_|2_|4_|5_|7_|13_|14_)", regex=True)) &
    (~ecb["Name"].str.startswith("-1_"))
]
ecb_mp.reset_index(drop=True, inplace=True)

ecbmp_docs = ecb_mp['summary'].tolist()
ecbmp_embeddings = ecb_mp['embeddings'].tolist()
ecbmp_embeddings = np.array(ecbmp_embeddings)

topic_model, topics, probabilities = bertopic_modeling(
    docs=ecbmp_docs,
    embeddings=ecbmp_embeddings,
    n_gram_range=(1, 2),
    min_cluster_size=10,
    umap_n_neighbors=3,
    umap_n_components=20,
    umap_min_dist=0.1,
    umap_metric='euclidean',
    umap_random_state=61,
)

df_topic_ecb_mp = topic_model.get_topic_info()
reduced_embeddings = UMAP(n_components=2, n_neighbors=20, min_dist=0.5, random_state=42).fit_transform(ecbmp_embeddings)
topic_model.visualize_documents(ecbmp_docs, reduced_embeddings=reduced_embeddings, hide_document_hover=True, hide_annotations=True)

df_reduced_embeddings = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
ecb_mp['x'] = df_reduced_embeddings['x']
ecb_mp['y'] = df_reduced_embeddings['y']

ctfidf_matrix = topic_model.c_tf_idf_
feature_names = topic_model.vectorizer_model.get_feature_names_out()
ctfidf_dense = ctfidf_matrix.todense()
ctfidf_df = pd.DataFrame(ctfidf_dense, columns=feature_names)

top_30_df = pd.DataFrame(columns=["Keyword_Ranking"])
for i in range(len(topic_model.generate_topic_labels())):
    top_terms = ctfidf_df.loc[i].sort_values(ascending=False).head(30)
    lst_top_terms = top_terms.index.tolist()
    top_30_df = pd.concat([top_30_df, pd.DataFrame({"topic_vector": [lst_top_terms]})], ignore_index=True)
    
df_topic_ecb_mp = pd.merge(df_topic_ecb_mp, top_30_df, how='left', left_index=True, right_index=True)
df_res = pd.DataFrame({'Topic': topics, 'Probability': probabilities})
df_res = df_res.merge(df_topic_ecb_mp, how='left', on='Topic')



ecb_mp = ecb_mp.drop(columns=['Topic', 'Probability', 'Count', 'Name',
                              'Representation', 'Representative_Docs', 'Keyword_Ranking',
                              'topic_vector'])

ecb_mp = pd.merge(ecb_mp, df_res, left_index=True, right_index=True, how='left')

fig = px.scatter_3d(
    ecb_mp,
    x='x',
    y='date',
    z='y',
    color='Name',
    title='Interactive 3D Scatter Plot',
    hover_data=['title', 'Name']
)
fig.update_layout(legend_title_text='Name')
fig.write_html(f'{output_dir_plots}/3d_plot_ecb_mp.html')

plot_data = ecb_mp.copy()
plot_data['date'] = pd.to_datetime(plot_data['date'])

unique_names = plot_data['Name'].unique()
color_map = cm.get_cmap('Paired', len(unique_names))
name_to_color = {name: color_map(i) for i, name in enumerate(unique_names)}

custom_colors = [
    "#1f77b4",
    "#ff7f0e",
    "#E1E1E1",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#4daf4a",
    "#d62728",
    "#17becf"
]

name_to_color = {name: custom_colors[i % len(custom_colors)] for i, name in enumerate(unique_names)}
plot_data['date_num'] = date2num(plot_data['date'])

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for name in unique_names:
    subset = plot_data[plot_data['Name'] == name]
    ax.plot(
        subset['x'], subset['date_num'], subset['y'],
        linestyle='', marker='o', markersize=10,
        color=name_to_color[name], alpha=0.7
    )

ax.set_xlabel('')
ax.set_ylabel('Year')
ax.set_zlabel('')
ax.view_init(elev=40, azim=180)
ax.invert_yaxis()
ax.set_xticklabels([])
ax.set_zticklabels([])

date_labels = [num2date(d).strftime('%Y') for d in ax.get_yticks()]
ax.set_yticklabels(date_labels)

ax.grid(True)

plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
plt.tight_layout()

pdf_path = f'{output_dir_plots}/ecb_mp_3d.pdf'
png_path = f'{output_dir_plots}/ecb_mp_3d.png'
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

print("Saved to", pdf_path)

file_path = "euro_area_cpi.csv"  # Path to your local file
df = pd.read_csv(file_path, index_col=0, parse_dates=True)  # Load the data, parsing the index as dates


df['YoY Change (%)'] = df["Euro_Area_CPI"].pct_change(periods=12) * 100
df = df.dropna()

fig, ax = plt.subplots(figsize=(9, 9))
ax.plot(df.index, df['YoY Change (%)'], label="YoY Change (Euro Area CPI)", color="blue")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator(5))
fig.autofmt_xdate()
ax.set_xlabel("Year")
ax.set_ylabel("YoY Change (%)")
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.grid(True)
plt.tight_layout()


pdf_path = f'{output_dir_plots}/ecb_inflation.pdf'
png_path = f'{output_dir_plots}/ecb_inflation.png'
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

print("Saved to", pdf_path)