import os
import xmltodict
import pandas as pd

# === 1. Recursively collect all XML file paths ===
def get_all_xml_paths(base_path):
    xml_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".xml"):
                xml_files.append(os.path.join(root, file))
    return xml_files

# === 2. Parse one XML file and extract speaker/party/text ===
def parse_speeches(xml_path):
    with open(xml_path, 'r', encoding='utf-8') as file:
        xml_data = xmltodict.parse(file.read())

    speeches = []
    try:
        divs = xml_data["teiCorpus"]["TEI"]["text"]["body"]["div"]
        if isinstance(divs, dict):  # only one debate
            divs = [divs]
    except:
        return speeches  # skip broken file

    for div in divs:
        if "sp" not in div:
            continue
        speeches_in_div = div["sp"]
        if isinstance(speeches_in_div, dict):
            speeches_in_div = [speeches_in_div]

        for sp in speeches_in_div:
            speaker = sp.get("@who_original", "")
            party = sp.get("@party", "")
            paragraphs = sp.get("p", [])

            if isinstance(paragraphs, str):
                paragraphs = [paragraphs]
            elif isinstance(paragraphs, dict):
                paragraphs = [paragraphs.get("#text", "")]

            text = " ".join(paragraphs)
            if speaker and text.strip():
                speeches.append({
                    "speaker": speaker,
                    "party": party,
                    "text": text
                })

    return speeches

# === 3. Process all XML files ===
# Update this path if needed — relative to where the script is run from
data_dir = "Data"  # or "../Data" if you're running from inside /scripts

all_speeches = []
all_files = get_all_xml_paths(data_dir)
print(f"Found {len(all_files)} XML files.")

for path in all_files:
    speeches = parse_speeches(path)
    all_speeches.extend(speeches)

# === 4. Convert to DataFrame and filter Merkel & SPD ===
df = pd.DataFrame(all_speeches)
df_merkel = df[df["speaker"].str.contains("Merkel", case=False)]
df_spd = df[(df["party"] == "SPD") & (~df["speaker"].str.contains("Merkel", case=False))]

print(f"Merkel speeches: {len(df_merkel)}")
print(f"SPD speeches: {len(df_spd)}")

# === 5. Save CSVs (optional) ===
df_merkel.to_csv("data/filtered/df_merkel.csv", index=False)
df_spd.to_csv("data/filtered/df_spd.csv", index=False)

