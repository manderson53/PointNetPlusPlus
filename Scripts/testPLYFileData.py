from plyfile import PlyData

def inspect_ply_channels(ply_path):
    plydata = PlyData.read(ply_path)
    # PLY files store vertices as an element, find it
    vertex_element = None
    for element in plydata.elements:
        if element.name == 'vertex':
            vertex_element = element
            break
    if vertex_element is None:
        print("No vertex element found in PLY file.")
        return

    print(f"Properties (channels) in vertex element of '{ply_path}':")
    for prop in vertex_element.properties:
        print(f"- {prop.name}")

if __name__ == "__main__":
    # Example PLY file path (replace with your own)
    example_ply_path = r"C:\Users\RemoteCollabHoloLens\Desktop\PointNet++\Benchmark\Benchmark\training_10_classes\Lille1_1.ply"
    inspect_ply_channels(example_ply_path)
