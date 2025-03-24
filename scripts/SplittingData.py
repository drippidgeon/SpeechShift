import os
import xmltodict
import pandas as pd

def parse_speeches(xml_path):
    with open(xml_path, 'r', encoding='utf-8') as file:
        xml_data = xmltodict.parse(file.read())

    speeches = []

    try:
        divs = xml_data["teiCorpus"]["TEI"]["text"]["body"]["div"]
        if isinstance(divs, dict):  # nur ein Redeblock
            divs = [divs]
    except:
        return speeches  # skip kaputte Datei

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
                paragraphs = [paragraphs["#text"]]

            text = " ".join(paragraphs)
            if speaker and text.strip():
                speeches.append({
                    "speaker": speaker,
                    "party": party,
                    "text": text
                })

    return speeches

# === Datenverarbeitung ===
data_dir = "data/"  # Dein Pfad zu den XML-Dateien
all_speeches = []

for file in os.listdir(data_dir):
    if file.endswith(".xml"):
        speeches = parse_speeches(os.path.join(data_dir, file))
        all_speeches.extend(speeches)

df = pd.DataFrame(all_speeches)

# === Filtere Merkel- und SPD-Reden ===
df_merkel = df[df["speaker"].str.contains("Merkel", case=False)]
df_spd = df[(df["party"] == "SPD") & (~df["speaker"].str.contains("Merkel", case=False))]

# Speichern als CSV (optional)
df_merkel.to_csv("data/df_merkel.csv", index=False)
df_spd.to_csv("data/df_spd.csv", index=False)
