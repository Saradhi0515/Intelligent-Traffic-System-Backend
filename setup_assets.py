import cv2
import numpy as np
import os

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_image(path, color, size):
    # Create a solid color image
    img = np.zeros((size[1], size[0], 3), np.uint8)
    img[:] = color
    cv2.imwrite(path, img)

def create_signal(path, color, size):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    # Draw a circle
    center = (size[0]//2, size[1]//2)
    radius = min(size[0], size[1]) // 2 - 2
    cv2.circle(img, center, radius, color, -1)
    cv2.imwrite(path, img)

def main():
    base_dir = "d:/Projects/Intelligent-Traffic-System/Intelligent-Traffic-System-Backend/Signal-Control/images"
    
    # Create directories
    create_dir(base_dir)
    create_dir(os.path.join(base_dir, "signals"))
    directions = ["right", "down", "left", "up"]
    for d in directions:
        create_dir(os.path.join(base_dir, d))

    # 1. Background
    # 1400x800 gray background
    bg = np.zeros((800, 1400, 3), np.uint8)
    bg[:] = (50, 50, 50) # Dark gray
    # Draw intersection lines (simple cross)
    cv2.line(bg, (500, 0), (500, 800), (255, 255, 255), 2)
    cv2.line(bg, (900, 0), (900, 800), (255, 255, 255), 2)
    cv2.line(bg, (0, 300), (1400, 300), (255, 255, 255), 2)
    cv2.line(bg, (0, 500), (1400, 500), (255, 255, 255), 2)
    cv2.imwrite(os.path.join(base_dir, "mod_int.png"), bg)

    # 2. Signals
    # Red, Yellow, Green
    # Size approx 30x30? The code blits them at signalCoods.
    # Let's make them 40x40
    create_signal(os.path.join(base_dir, "signals", "red.png"), (0, 0, 255), (40, 40))
    create_signal(os.path.join(base_dir, "signals", "yellow.png"), (0, 255, 255), (40, 40))
    create_signal(os.path.join(base_dir, "signals", "green.png"), (0, 255, 0), (40, 40))

    # 3. Vehicles
    # car, bus, truck, rickshaw, bike
    # Sizes need to be somewhat realistic relative to the road
    # Car: Blue, 40x20 (horizontal) / 20x40 (vertical)
    # But the code rotates them? 
    # Line 118: path = "images/" + direction + "/" + vehicleClass + ".png"
    # So we need pre-rotated images for each direction.
    
    vehicles = {
        "car": {"color": (255, 0, 0), "size": (40, 20)},      # Blue
        "bus": {"color": (0, 255, 255), "size": (80, 25)},    # Yellow
        "truck": {"color": (255, 0, 255), "size": (80, 25)},  # Magenta
        "rickshaw": {"color": (0, 165, 255), "size": (25, 15)}, # Orange-ish
        "bike": {"color": (0, 255, 0), "size": (20, 10)}      # Green
    }

    for d in directions:
        for v_name, v_props in vehicles.items():
            w, h = v_props["size"]
            color = v_props["color"]
            
            # Adjust dimensions based on direction
            # right/left: horizontal (w, h)
            # up/down: vertical (h, w)
            if d in ["up", "down"]:
                size = (h, w)
            else:
                size = (w, h)
                
            path = os.path.join(base_dir, d, f"{v_name}.png")
            create_image(path, color, size)

    print("Assets generated successfully.")

if __name__ == "__main__":
    main()
