import json, sys
sys.path.insert(0, r"D:\WILP\Workingcode\Baseline\WILP")

json_path = r"d:\WILP\sem-4\Dataset\DENTEX\DENTEXsample2\training_data\quadrant-enumeration-disease\train_quadrant_enumeration_disease.json"
with open(json_path) as f:
    data = json.load(f)

# Build lookup tables (category_id → actual value) — IDs are 0-indexed, names are the real values
cat1_map = {cat["id"]: int(cat["name"]) for cat in data["categories_1"]}  # id → quadrant (1-4)
cat2_map = {cat["id"]: int(cat["name"]) for cat in data["categories_2"]}  # id → position (1-8)
cat3_map = {cat["id"]: cat["name"]      for cat in data["categories_3"]}  # id → disease name

for fname in ["train_10.png", "train_16.png"]:
    img = next((i for i in data["images"] if i["file_name"] == fname), None)
    if not img:
        print(f"{fname}: NOT FOUND")
        continue
    anns = [a for a in data["annotations"] if a["image_id"] == img["id"]]
    print(f"\n{fname}  ({img['width']}x{img['height']})  diseased_annotations={len(anns)}")
    print(f"  {'FDI':>5}  {'Quadrant':<12}  {'Position':<25}  Disease")
    print(f"  {'-'*5}  {'-'*12}  {'-'*25}  {'-'*20}")
    for a in anns:
        quad = cat1_map.get(a["category_id_1"], f"?id={a['category_id_1']}")
        pos  = cat2_map.get(a["category_id_2"], f"?id={a['category_id_2']}")
        dis  = cat3_map.get(a.get("category_id_3"), f"?id={a.get('category_id_3')}")
        fdi  = quad * 10 + pos
        quad_names = {1:"Upper Right", 2:"Upper Left", 3:"Lower Left", 4:"Lower Right"}
        pos_names  = {1:"Central Incisor",2:"Lateral Incisor",3:"Canine",
                      4:"First Premolar",5:"Second Premolar",6:"First Molar",
                      7:"Second Molar",8:"Wisdom Tooth"}
        print(f"  FDI {fdi:>2}  {quad_names.get(quad,'?'):<12}  {pos_names.get(pos,'?'):<25}  {dis}")
    print()
    print("  NOTE: category_id_1/2 are 0-indexed lookup IDs, NOT direct quadrant/position numbers.")
    print("  e.g. category_id_1=2 → categories_1 id=2 name='3' → quadrant 3 (Lower Left)")
