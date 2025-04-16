import os
import xmltodict
import pandas as pd

os.makedirs("../data/filtered", exist_ok=True)

def get_all_xml_paths(base_path):
    xml_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".xml"):
                xml_files.append(os.path.join(root, file))
    return xml_files

def parse_speeches(xml_path):
    speeches = []
    try:
        with open(xml_path, 'r', encoding='utf-8') as file:
            xml_data = xmltodict.parse(file.read())
    except Exception as e:
        print(f"ERROR parsing {xml_path}: {e}")
        return speeches

    try:
        divs = xml_data["TEI"]["text"]["body"]["div"]
        if isinstance(divs, dict):  # Single debate scenario
            divs = [divs]
    except Exception as e:
        print(f"SKIPPED {xml_path} – couldn't find div: {e}")
        return speeches

    for div in divs:
        if "sp" not in div:
            continue

        speakers = div["sp"]
        if isinstance(speakers, dict):
            speakers = [speakers]

        for sp in speakers:
            speaker = sp.get("@who_original", "") or sp.get("@name", "")
            party = sp.get("@party", "")
# Only extracts content from <p> = only speeches and no interjections
            paragraphs = sp.get("p", [])
            if isinstance(paragraphs, str):
                paragraphs = [paragraphs]
            elif isinstance(paragraphs, dict):
                paragraphs = [paragraphs.get("#text", "")]
            elif not isinstance(paragraphs, list):
                paragraphs = []

            text = " ".join(paragraphs).strip()
            if speaker and text:
                speeches.append({
                    "speaker": speaker,
                    "party": party,
                    "text": text
                })

    return speeches

data_dir = "../data/raw"
all_speeches = []
all_files = get_all_xml_paths(data_dir)
print(f"Found {len(all_files)} XML files.")

for path in all_files:
    speeches = parse_speeches(path)
    all_speeches.extend(speeches)

df = pd.DataFrame(all_speeches)
print(f"Total speeches collected: {len(df)}")

# Filter for Merkel and SPD speeches
df_merkel = df[df["speaker"].str.contains("Angela Merkel", case=False, na=False)]
df_spd = df[(df["party"] == "SPD") & (~df["speaker"].str.contains("Angel Merkel", case=False, na=False))]

print(f"Merkel speeches: {len(df_merkel)}")
print(f"SPD speeches (excluding Merkel): {len(df_spd)}")

# Save filtered data
df_merkel.to_csv("data/filtered/df_merkel.csv", index=False)
df_spd.to_csv("data/filtered/df_spd.csv", index=False)
print("Filtered datasets saved.")
