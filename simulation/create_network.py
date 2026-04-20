"""
create_network.py — Build Vadodara-focused SUMO network
10 real roads with actual lengths and connectivity.
"""
import subprocess, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Node definitions (junctions) ──────────────────────────
# 6 intersections + 3 terminal nodes
# Coordinates spread out for clear SUMO GUI visualization
nodes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <!-- Main intersections (traffic lights) -->
    <node id="natubhai"   x="0"     y="0"     type="traffic_light"/>
    <node id="prodmore"   x="0"     y="300"   type="traffic_light"/>
    <node id="karelibaug" x="250"   y="0"     type="traffic_light"/>
    <node id="sama"       x="550"   y="0"     type="traffic_light"/>
    <node id="raopura_jn" x="-200"  y="0"     type="traffic_light"/>
    <node id="manjalpur"  x="-200"  y="-300"  type="traffic_light"/>

    <!-- Terminal nodes -->
    <node id="gotri_end"    x="550"   y="550"   type="priority"/>
    <node id="newsama_end"  x="800"   y="0"     type="priority"/>
    <node id="oldpadra_end" x="-200"  y="-600"  type="priority"/>
</nodes>
"""

# ── Edge definitions (roads) ─────────────────────────────
# 10 Vadodara roads, bidirectional, real lengths, 2 lanes
edges_xml = """<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <!-- Race Course Road: Natubhai Circle → Prodmore Circle -->
    <edge id="racecourse_n" from="natubhai"   to="prodmore"   numLanes="2" speed="13.89" length="280.5"/>
    <edge id="racecourse_s" from="prodmore"   to="natubhai"   numLanes="2" speed="13.89" length="280.5"/>

    <!-- RC Dutt Road: Prodmore → Karelibaug -->
    <edge id="rcdutt_e"     from="prodmore"   to="karelibaug" numLanes="2" speed="13.89" length="291.9"/>
    <edge id="rcdutt_w"     from="karelibaug" to="prodmore"   numLanes="2" speed="13.89" length="291.9"/>

    <!-- Dandia Bazaar Road: Natubhai Circle → Karelibaug -->
    <edge id="dandia_e"     from="natubhai"   to="karelibaug" numLanes="2" speed="11.11" length="96.3"/>
    <edge id="dandia_w"     from="karelibaug" to="natubhai"   numLanes="2" speed="11.11" length="96.3"/>

    <!-- Jetalpur Road: Karelibaug → Sama -->
    <edge id="jetalpur_e"   from="karelibaug" to="sama"       numLanes="2" speed="13.89" length="199.0"/>
    <edge id="jetalpur_w"   from="sama"       to="karelibaug" numLanes="2" speed="13.89" length="199.0"/>

    <!-- Gotri Road: Sama → Gotri End -->
    <edge id="gotri_n"      from="sama"       to="gotri_end"  numLanes="2" speed="13.89" length="505.8"/>
    <edge id="gotri_s"      from="gotri_end"  to="sama"       numLanes="2" speed="13.89" length="505.8"/>

    <!-- New Sama Road: Sama → New Sama End -->
    <edge id="newsama_e"    from="sama"       to="newsama_end" numLanes="2" speed="13.89" length="188.4"/>
    <edge id="newsama_w"    from="newsama_end" to="sama"       numLanes="2" speed="13.89" length="188.4"/>

    <!-- Raopura Road: Natubhai Circle → Raopura Junction -->
    <edge id="raopura_w"    from="natubhai"   to="raopura_jn" numLanes="2" speed="11.11" length="93.7"/>
    <edge id="raopura_e"    from="raopura_jn" to="natubhai"   numLanes="2" speed="11.11" length="93.7"/>

    <!-- Manjalpur Gate Road: Raopura Jn → Manjalpur -->
    <edge id="manjalpur_s"  from="raopura_jn" to="manjalpur"  numLanes="2" speed="11.11" length="178.8"/>
    <edge id="manjalpur_n"  from="manjalpur"  to="raopura_jn" numLanes="2" speed="11.11" length="178.8"/>

    <!-- Old Padra Road: Manjalpur → Old Padra End -->
    <edge id="oldpadra_s"   from="manjalpur"  to="oldpadra_end" numLanes="2" speed="13.89" length="264.7"/>
    <edge id="oldpadra_n"   from="oldpadra_end" to="manjalpur"  numLanes="2" speed="13.89" length="264.7"/>

    <!-- Natubhai Circle connector (very short, models the roundabout) -->
    <edge id="natubhai_loop" from="natubhai"  to="raopura_jn" numLanes="1" speed="8.33"  length="30.4"
          allow="passenger"/>
</edges>
"""

# Write XML files
with open("vadodara.nod.xml", "w", encoding="utf-8") as f:
    f.write(nodes_xml)
with open("vadodara.edg.xml", "w", encoding="utf-8") as f:
    f.write(edges_xml)

print("Building Vadodara network...")

# Find netconvert
sumo_home = os.environ.get("SUMO_HOME", r"D:\Program Files\sumosimulator")
netconvert = os.path.join(sumo_home, "bin", "netconvert")
if not os.path.exists(netconvert):
    netconvert = "netconvert"  # fallback to PATH

result = subprocess.run([
    netconvert,
    "--node-files", "vadodara.nod.xml",
    "--edge-files", "vadodara.edg.xml",
    "--output-file", "network.net.xml",
    "--no-turnarounds", "true",
    "--junctions.join", "true",
], capture_output=True, text=True)

if result.returncode == 0:
    print("✓ Network created: network.net.xml")
    print("  9 junctions (6 traffic lights)")
    print("  19 edges (10 Vadodara roads, bidirectional)")
else:
    print(f"Error: {result.stderr}")