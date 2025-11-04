import json
import os

# Percorsi dei file esistenti
train_json = "C:/Tirocinio/DEIMv2/dataset/annotations/instances_train.json"
val_json   = "C:/Tirocinio/DEIMv2/dataset/annotations/instances_val.json"

# Percorso file di output
trainval_json = "C:/Tirocinio/DEIMv2/dataset/annotations/trainval.json"

# Carica train
with open(train_json, "r") as f:
    train_data = json.load(f)

# Carica val
with open(val_json, "r") as f:
    val_data = json.load(f)

# Unisci immagini e annotazioni
images = train_data["images"] + val_data["images"]
annotations = train_data["annotations"] + val_data["annotations"]

# Le categorie dovrebbero essere identiche in train e val
categories = train_data.get("categories", val_data.get("categories", []))

# Crea dizionario finale
trainval_data = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

# Salva il file trainval.json
os.makedirs(os.path.dirname(trainval_json), exist_ok=True)
with open(trainval_json, "w") as f:
    json.dump(trainval_data, f)

print(f"File trainval.json creato correttamente in: {trainval_json}")
