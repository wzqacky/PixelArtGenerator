import io
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def build_canny_map(src_path, src_byte=None, plot=False, return_byte=True):
    if src_byte:
        try:
            image_buffer = np.frombuffer(src_byte, np.uint8)
            src_img = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error reading and decoding bytes for {src_path}.")
    else:
        src_img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian filter for blurring
    blur_img = cv2.GaussianBlur(img, (7,7), 0)
    # Gradient filters
    gx = cv2.Sobel(np.float32(blur_img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(blur_img), cv2.CV_64F, 0, 1, 3)
    mag = np.sqrt(gx**2 + gy**2)
    max_mag = np.max(mag)
    T_lower, T_upper = max_mag * 0.1, max_mag * 0.5
    edges = cv2.Canny(blur_img, T_lower, T_upper)
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        ax1.imshow(src_img)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Plot Canny edges
        ax2.imshow(edges, cmap='gray')
        ax2.set_title('Canny Edges')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(f"visualization/canny/{'-'.join(src_path.split('/'))}_canny.png")
    if not return_byte:
        return Image.fromarray(edges)
    # Encode the edge map to PNG bytes
    is_success, buffer = cv2.imencode(".png", edges)
    if not is_success:
        print(f"Warning: Could not encode canny edge for {src_path}. Skipping.")
        return None
    return buffer.tobytes()

def build_color_palette(src_path, src_byte=None, plot=False, return_byte=True):
    if src_byte:
        try:
            src_img = Image.open(io.BytesIO(src_byte))
        except Exception as e:
            print(f"Error reading and decoding bytes for {src_path}.")
    else:
        src_img = Image.open(src_path)
    original_size = src_img.size
    # Downsample for pixelation effect
    pixel_size = 16
    pixelated_width = original_size[0] // pixel_size
    pixelated_height = original_size[1] // pixel_size
    if pixelated_width < 1: small_width = 1
    if pixelated_height < 1: small_height = 1

    pixelated_img = src_img.resize((pixelated_width, pixelated_height), Image.Resampling.LANCZOS)
    # Find dominant colors for the flattened pixels
    n_colors = 2
    pixels = np.array(pixelated_img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto').fit(pixels)
    
    # Map each pixel to the closest palette color to form the 2D palette
    labels = kmeans.predict(pixels)
    new_pixels = kmeans.cluster_centers_[labels].astype('uint8')
    new_pixelated_data = new_pixels.reshape(pixelated_height, pixelated_width, 3)
    new_pixelated_img = Image.fromarray(new_pixelated_data, 'RGB')
    
    # Scale back to original size
    final_img = new_pixelated_img.resize(original_size, Image.Resampling.NEAREST)
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.imshow(src_img)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Plot Canny edges
        ax2.imshow(final_img)
        ax2.set_title('Color Palette')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(f"visualization/palette/{'-'.join(src_path.split('/'))}_palette.png")
    if not return_byte:
        return final_img
    output_buffer = io.BytesIO()
    final_img.save(output_buffer, format='png')
    return output_buffer.getvalue()
