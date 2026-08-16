"""
Self-estimated textual frequency distributions: when an English speaker
encounters WORD in isolation, how likely is each sense?
Estimates drawn from Claude's training on internet-scale text.

Writes the manual prior to freq_manual.json and validates each entry against the
senses listed in meanings.json (warns on any word/sense mismatch).

    python -m llm_polysemy.freq.build_freq
"""
import json
from pathlib import Path

from ..meanings import load_meanings

OUT_FILE = Path("data/freq_manual.json")

# fmt: off
PRIORS = {
    "trunk":    {"tree": 0.30, "car": 0.30, "elephant": 0.25, "body": 0.05, "luggage": 0.10},
    "jordan":   {"country": 0.35, "person": 0.55, "shoes": 0.10},
    "bolt":     {"fastener": 0.40, "lightning": 0.40, "run": 0.10, "lock": 0.10},
    "bank":     {"finance": 0.70, "river": 0.25, "tilt": 0.03, "heap": 0.02},
    "pitch":    {"throw": 0.20, "tone": 0.20, "sales": 0.25, "field": 0.35},
    "bow":      {"bend": 0.20, "weapon": 0.35, "knot": 0.25, "ship": 0.10, "violin": 0.10},
    "spring":   {"season": 0.60, "coil": 0.15, "water": 0.15, "leap": 0.10},
    "crane":    {"bird": 0.30, "machine": 0.60, "stretch": 0.10},
    "club":     {"organization": 0.30, "venue": 0.25, "weapon": 0.10, "golf": 0.25, "cards": 0.10},
    "turkey":   {"bird": 0.20, "country": 0.30, "food": 0.35, "bowling": 0.05, "failure": 0.10},
    "shot":     {"gunfire": 0.15, "injection": 0.15, "attempt": 0.25, "drink": 0.20, "photo": 0.25},
    "bar":      {"pub": 0.35, "rod": 0.20, "block": 0.10, "law": 0.20, "music": 0.15},
    "mercury":  {"planet": 0.35, "element": 0.45, "god": 0.20},
    "jack":     {"lift": 0.25, "cards": 0.25, "name": 0.35, "connector": 0.15},
    "mole":     {"animal": 0.30, "skin": 0.35, "spy": 0.15, "chemistry": 0.15, "sauce": 0.05},
    "charge":   {"electric": 0.25, "fee": 0.20, "rush": 0.15, "accusation": 0.20, "responsibility": 0.20},
    "key":      {"lock": 0.30, "crucial": 0.30, "music": 0.15, "keyboard": 0.15, "legend": 0.10},
    "watch":    {"timepiece": 0.40, "observe": 0.50, "guard": 0.10},
    "ring":     {"jewelry": 0.35, "sound": 0.20, "circle": 0.15, "arena": 0.15, "call": 0.15},
    "nail":     {"finger": 0.40, "fastener": 0.45, "succeed": 0.15},
    "chip":     {"fragment": 0.15, "microchip": 0.30, "snack": 0.40, "token": 0.15},
    "cell":     {"biology": 0.25, "prison": 0.15, "phone": 0.35, "battery": 0.10, "spreadsheet": 0.15},
    "toast":    {"bread": 0.45, "tribute": 0.40, "doomed": 0.15},
    "track":    {"path": 0.15, "railroad": 0.15, "song": 0.30, "follow": 0.20, "athletics": 0.20},
    "wave":     {"water": 0.40, "gesture": 0.30, "physics": 0.20, "surge": 0.10},
    "train":    {"railway": 0.60, "teach": 0.25, "gown": 0.05, "sequence": 0.10},
    "anchor":   {"ship": 0.35, "news": 0.45, "secure": 0.20},
    "ash":      {"residue": 0.65, "tree": 0.35},
    "ball":     {"sphere": 0.55, "dance": 0.25, "fun": 0.20},
    "band":     {"music": 0.45, "strip": 0.25, "unite": 0.15, "frequency": 0.15},
    "bark":     {"dog": 0.40, "tree": 0.50, "ship": 0.10},
    "barrel":   {"cask": 0.35, "gun": 0.20, "oil": 0.45},
    "basket":   {"container": 0.65, "basketball": 0.35},
    "bass":     {"sound": 0.30, "fish": 0.25, "instrument": 0.45},
    "batter":   {"baseball": 0.35, "cooking": 0.45, "beat": 0.20},
    "beam":     {"light": 0.35, "support": 0.35, "smile": 0.20, "transmit": 0.10},
    "bench":    {"seat": 0.40, "sports": 0.25, "court": 0.20, "workbench": 0.15},
    "block":    {"solid": 0.25, "obstruct": 0.30, "city": 0.45},
    "bowl":     {"dish": 0.50, "roll": 0.20, "stadium": 0.30},
    "box":      {"container": 0.60, "fight": 0.25, "rectangle": 0.15},
    "brush":    {"tool": 0.55, "touch": 0.30, "shrubs": 0.15},
    "buck":     {"deer": 0.25, "dollar": 0.55, "resist": 0.20},
    "button":   {"fastener": 0.35, "press": 0.45, "badge": 0.20},
    "cane":     {"stick": 0.40, "plant": 0.40, "beat": 0.20},
    "cape":     {"cloak": 0.35, "land": 0.65},
    "capital":  {"city": 0.45, "money": 0.25, "letter": 0.15, "punishment": 0.15},
    "case":     {"container": 0.15, "legal": 0.30, "instance": 0.35, "investigation": 0.20},
    "chest":    {"body": 0.55, "box": 0.45},
    "coach":    {"trainer": 0.45, "vehicle": 0.25, "tutor": 0.20, "class": 0.10},
    "cobbler":  {"shoemaker": 0.50, "dessert": 0.50},
    "apple":    {"fruit": 0.30, "company": 0.70},
    "amazon":   {"river": 0.10, "rainforest": 0.15, "company": 0.70, "warrior": 0.05},
    "mars":     {"planet": 0.45, "god": 0.20, "candy": 0.35},
    "jaguar":   {"animal": 0.45, "car": 0.55},
    "mustang":  {"horse": 0.35, "car": 0.65},
    "cobra":    {"snake": 0.85, "yoga": 0.15},
    "python":   {"snake": 0.30, "language": 0.70},
    "fox":      {"animal": 0.45, "cunning": 0.15, "network": 0.40},
    "cardinal": {"bird": 0.35, "clergy": 0.35, "main": 0.20, "number": 0.10},
    "ram":      {"sheep": 0.30, "force": 0.20, "memory": 0.50},
    "paddle":   {"oar": 0.50, "bat": 0.25, "wade": 0.25},
    "staple":   {"fastener": 0.40, "essential": 0.60},
    "pitcher":  {"baseball": 0.50, "jug": 0.50},
    "shuttle":  {"transport": 0.35, "spacecraft": 0.40, "badminton": 0.15, "weaving": 0.10},
    "tap":      {"faucet": 0.30, "hit": 0.30, "dance": 0.20, "access": 0.20},
    "drill":    {"tool": 0.45, "exercise": 0.35, "military": 0.20},
    "pump":     {"device": 0.50, "shoe": 0.25, "inflate": 0.25},
    "socket":   {"outlet": 0.45, "joint": 0.25, "tool": 0.30},
    "switch":   {"control": 0.45, "change": 0.45, "rod": 0.10},
    "iron":     {"metal": 0.35, "appliance": 0.30, "press": 0.10, "golf": 0.25},
    "crown":    {"headwear": 0.30, "top": 0.20, "monarchy": 0.50},
    "scale":    {"weigh": 0.25, "skin": 0.15, "music": 0.15, "size": 0.30, "climb": 0.15},
    "plane":    {"aircraft": 0.65, "surface": 0.20, "tool": 0.15},
    "fan":      {"cooling": 0.30, "admirer": 0.55, "spread": 0.15},
    "bridge":   {"crossing": 0.55, "card game": 0.20, "nose": 0.15, "ship": 0.10},
    "port":     {"harbor": 0.35, "left": 0.15, "wine": 0.20, "connector": 0.30},
    "lodge":    {"cabin": 0.55, "file": 0.25, "stuck": 0.20},
    "mine":     {"excavation": 0.25, "explosive": 0.25, "possessive": 0.50},
    "grave":    {"burial": 0.50, "serious": 0.50},
    "cast":     {"actors": 0.35, "throw": 0.20, "medical": 0.35, "mold": 0.10},
    "press":    {"media": 0.45, "push": 0.35, "printing": 0.20},
    "post":     {"mail": 0.25, "pole": 0.15, "job": 0.20, "publish": 0.25, "station": 0.15},
    "tip":      {"gratuity": 0.30, "point": 0.25, "advice": 0.30, "tilt": 0.15},
    "will":     {"volition": 0.15, "testament": 0.15, "future": 0.70},
    "file":     {"document": 0.50, "submit": 0.20, "tool": 0.15, "line": 0.15},
    "mortar":   {"cement": 0.45, "bowl": 0.20, "weapon": 0.35},
    "hide":     {"conceal": 0.55, "skin": 0.45},
    "tank":     {"container": 0.30, "vehicle": 0.55, "fail": 0.15},
    "seal":     {"animal": 0.35, "close": 0.40, "stamp": 0.25},
    "strike":   {"hit": 0.30, "labor": 0.30, "baseball": 0.20, "bowling": 0.20},
    "stock":    {"shares": 0.40, "supply": 0.20, "broth": 0.20, "livestock": 0.20},
    "bug":      {"insect": 0.30, "defect": 0.40, "annoy": 0.20, "surveillance": 0.10},
    "fly":      {"insect": 0.35, "travel": 0.50, "zipper": 0.15},
    "bat":      {"animal": 0.35, "sports": 0.50, "hit": 0.15},
    "pen":      {"writing": 0.60, "enclosure": 0.20, "prison": 0.20},
    "pool":     {"swimming": 0.45, "billiards": 0.20, "shared": 0.25, "puddle": 0.10},
    "rock":     {"stone": 0.40, "music": 0.45, "sway": 0.15},
    "fire":     {"flames": 0.55, "dismiss": 0.25, "shoot": 0.20},
    "match":    {"firestick": 0.25, "contest": 0.45, "pairing": 0.30},
    "tie":      {"necktie": 0.40, "fasten": 0.15, "draw": 0.25, "bond": 0.20},
}
# fmt: on

def main():
    # Sanity check: all distributions should sum to ~1
    for word, dist in PRIORS.items():
        s = sum(dist.values())
        if abs(s - 1.0) > 0.01:
            print(f"WARNING {word} sums to {s:.3f}")

    # Validate every prior word/sense against meanings.json
    meanings = load_meanings()
    for word, dist in PRIORS.items():
        if word not in meanings:
            print(f"WARNING prior word {word!r} not in meanings.json")
            continue
        listed = {s["sense"] for s in meanings[word]}
        for sense in dist:
            if sense not in listed:
                print(f"WARNING {word}: prior sense {sense!r} not in meanings.json senses {sorted(listed)}")
    missing = [w for w in meanings if w not in PRIORS]
    if missing:
        print(f"NOTE {len(missing)} meanings words have no manual prior: {', '.join(missing[:10])}"
              + (" ..." if len(missing) > 10 else ""))

    OUT_FILE.write_text(json.dumps(PRIORS, indent=2))
    print(f"Saved {OUT_FILE} ({len(PRIORS)} words)")


if __name__ == "__main__":
    main()
