"""
Generate realistic vehicle routes for the Vadodara road network.
Uses only verified edge connections from the SUMO network.
"""
import os, random, xml.etree.ElementTree as ET

os.chdir(os.path.dirname(os.path.abspath(__file__)))
random.seed(42)

# All valid routes (verified against network connectivity)
routes = [
    # Gotri corridors
    "gotri_s jetalpur_w rcdutt_w racecourse_s",
    "gotri_s jetalpur_w dandia_w racecourse_n",
    "gotri_s jetalpur_w dandia_w raopura_w manjalpur_s oldpadra_s",
    "gotri_s newsama_e",
    # New Sama corridors
    "newsama_w jetalpur_w rcdutt_w racecourse_s",
    "newsama_w jetalpur_w dandia_w raopura_w manjalpur_s",
    "newsama_w gotri_n",
    # Old Padra corridors
    "oldpadra_n manjalpur_n raopura_e racecourse_n",
    "oldpadra_n manjalpur_n raopura_e dandia_e jetalpur_e gotri_n",
    "oldpadra_n manjalpur_n raopura_e dandia_e rcdutt_w",
    # Race Course corridors
    "racecourse_s dandia_e jetalpur_e gotri_n",
    "racecourse_s dandia_e jetalpur_e newsama_e",
    "racecourse_s raopura_w manjalpur_s oldpadra_s",
    "racecourse_s natubhai_loop raopura_e dandia_e",
    # RC Dutt corridors
    "rcdutt_e jetalpur_e gotri_n",
    "rcdutt_e jetalpur_e newsama_e",
    "rcdutt_e dandia_w raopura_w manjalpur_s oldpadra_s",
    "rcdutt_w racecourse_s raopura_w manjalpur_s",
    # Dandia corridors
    "dandia_e jetalpur_e gotri_n",
    "dandia_e rcdutt_w racecourse_s",
    "dandia_w racecourse_n rcdutt_e",
    "dandia_w raopura_w manjalpur_s oldpadra_s",
    # Raopura / Manjalpur
    "raopura_e racecourse_n rcdutt_e jetalpur_e",
    "raopura_e dandia_e jetalpur_e gotri_n",
    "manjalpur_n raopura_e racecourse_n",
    "manjalpur_s oldpadra_s",
    # Jetalpur
    "jetalpur_w dandia_w racecourse_n",
    "jetalpur_e gotri_n",
]

root = ET.Element("routes")

vtype = ET.SubElement(root, "vType",
                      id="car", accel="2.6", decel="4.5",
                      sigma="0.5", length="5", maxSpeed="13.89")

for i, edge_str in enumerate(routes):
    ET.SubElement(root, "route", id=f"r{i}", edges=edge_str)

vid = 0
for t in range(0, 3600):
    if t < 300:      rate = 0.3
    elif t < 1200:   rate = 0.8
    elif t < 2400:   rate = 1.0
    elif t < 3000:   rate = 0.7
    else:            rate = 0.4

    if random.random() < rate * 0.5:
        ET.SubElement(root, "vehicle",
                      id=f"v{vid}", type="car",
                      route=f"r{random.randint(0, len(routes)-1)}",
                      depart=str(t), departLane="best", departSpeed="max")
        vid += 1

tree = ET.ElementTree(root)
ET.indent(tree, space="    ")
tree.write("routes.rou.xml", encoding="unicode", xml_declaration=True)
print(f"✓ Routes: {vid} vehicles, {len(routes)} routes → routes.rou.xml")
