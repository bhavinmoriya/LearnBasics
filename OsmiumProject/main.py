import osmium


class MyHandler(osmium.SimpleHandler):
    def node(self, n):
        print(f"Node: {n.id}, Location: {n.location}")


class MyWriter(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.buffer = osmium.Buffer()

    def node(self, n):
        self.buffer.add_node(n)

    def write_to_file(self, filename):
        writer = osmium.osm.OsmWriter(filename)
        writer.write(self.buffer)
class HighwayHandler(osmium.SimpleHandler):
    def way(self, w):
        if "highway" in w.tags:
            print(f"Highway: {w.id}, Tags: {w.tags}")
class HistoryHandler(osmium.SimpleHandler):
    def node(self, n):
        print(f"Node {n.id} version {n.version}")

import geopandas as gpd
from osmium import SimpleHandler

class GeoHandler(SimpleHandler):
    def __init__(self):
        super().__init__()
        self.data = []

    def way(self, w):
        if "building" in w.tags:
            self.data.append({"id": w.id, "tags": w.tags})










def main():
    handler = MyHandler()
    handler.apply_file("esslingen.osm", locations=True)
    # Example usage
    writer = MyWriter()
    # Add nodes, ways, or relations to `writer.buffer`
    writer.write_to_file("output.osm.pbf")
    handler = HighwayHandler()
    handler.apply_file("esslingen.osm")

    handler = HistoryHandler()
    handler.apply_file("history.osh.pbf")
    handler = GeoHandler()
    handler.apply_file("esslingen.osm")
    gdf = gpd.GeoDataFrame(handler.data)


if __name__ == "__main__":
    main()
