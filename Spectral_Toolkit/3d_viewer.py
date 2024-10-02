import numpy as np
import plotly.graph_objects as go
from PIL import Image

obj_file_name = "C:/Users/tomas/Desktop/Rocha assets/rocha_spie.obj"
texture_file_name = "C:/Users/tomas/Desktop/Rocha assets/textures/Image_0.png"


def parse_obj_file(obj_file_path):
    vertices = []
    uvs = []
    faces = []
    face_uvs = []

    with open(obj_file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Vertex position
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif line.startswith('vt '):  # Texture coordinates (UV)
                parts = line.split()
                uvs.append([float(parts[1]), float(parts[2])])

            elif line.startswith('f '):  # Face definitions
                parts = line.split()[1:]
                face = []
                uv_face = []
                for part in parts:
                    indices = part.split('/')
                    face.append(int(indices[0]) - 1)  # Vertex index
                    uv_face.append(int(indices[1]) - 1)  # UV index
                faces.append(face)
                face_uvs.append(uv_face)

    return np.array(vertices), np.array(uvs), np.array(faces), np.array(face_uvs)


def interpolate_texture_color(uv_coords, texture_image):
    """
    Given UV coordinates, extract the corresponding color from the texture image.
    :param uv_coords: UV coordinates (2D)
    :param texture_image: Texture image (PIL Image object)
    :return: RGB color as [R, G, B]
    """
    # Convert UV coordinates (which are between 0 and 1) to pixel coordinates in the texture image
    width, height = texture_image.size
    u, v = uv_coords
    # Ensure the UV coordinates wrap around (UVs might be outside the 0-1 range)
    u = u % 1
    v = v % 1
    # Convert UV coordinates to pixel values
    pixel_x = int(u * width)
    pixel_y = int((1 - v) * height)  # Flip Y-axis as images have origin at the top-left
    # Get the color at the pixel location
    color = texture_image.getpixel((pixel_x, pixel_y))  # Returns a tuple (R, G, B)
    return color


# Load the OBJ file
vertices, uvs, faces, face_uvs = parse_obj_file(obj_file_name)

# Load the texture image
texture_image = Image.open(texture_file_name)

# Prepare the figure
fig = go.Figure()

# Prepare face colors by mapping UVs to texture colors
face_colors = []
for face, uv_indices in zip(faces, face_uvs):
    # Get the UV coordinates for this face
    uv_coords = [uvs[uv_index] for uv_index in uv_indices]
    
    # For simplicity, let's take the average UV coordinate for the face and map it to a color
    avg_uv = np.mean(uv_coords, axis=0)
    
    # Get the color from the texture
    color = interpolate_texture_color(avg_uv, texture_image)
    
    # Convert the RGB color (0-255 range) to normalized RGB (0-1 range) for plotly
    normalized_color = [c / 255.0 for c in color]
    
    # Append the color to the face_colors list (plotly uses single color per face for `Mesh3d`)
    face_colors.append(f"rgb({int(color[0])}, {int(color[1])}, {int(color[2])})")

# Create the mesh plot
fig.add_trace(go.Mesh3d(
    x=vertices[:, 0],
    y=vertices[:, 1],
    z=vertices[:, 2],
    i=faces[:, 0],
    j=faces[:, 1],
    k=faces[:, 2],
    facecolor=face_colors,  # Apply the face colors
    flatshading=True
))

# Show the plot
fig.show()