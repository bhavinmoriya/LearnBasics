from shapely.geometry import Point, LineString
from shapely import from_wkt, from_wkb
import binascii

# Create a point
point = Point(1.0, 2.0)

# WKT representation
wkt = point.wkt  # 'POINT (1.0000000000000000 2.0000000000000000)'

# WKB representation
wkb = point.wkb  # Binary string
wkb_hex = binascii.hexlify(wkb).decode('utf-8')  # '0101000000000000000000F03F0000000000000040'

print("WKT:", wkt)
print("WKB (hex):", wkb_hex)
print("From WKT:", from_wkt(wkt))
print("WKB (hex):", from_wkb(wkb))

