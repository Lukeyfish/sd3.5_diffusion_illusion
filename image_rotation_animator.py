from PIL import Image
import math
import numpy as np
from PIL import ImageDraw

class ImageAnimator:
    def __init__(self, image_path, num_divisions=4, frames_per_rotation=30):
        """
        Initialize the animator with the source image and animation parameters.
        
        Args:
            image_path: Path to the source image
            num_divisions: Number of divisions along each axis
            frames_per_rotation: Number of frames for a 90-degree rotation
        """
        self.image = Image.open(image_path).convert('RGBA')
        self.num_divisions = num_divisions
        self.frames_per_rotation = frames_per_rotation
        
    def _rotate_tile(self, tile, angle):
        """
        Rotate a tile around its center by the specified angle with a transparent buffer to reduce tearing.
        
        Args:
            tile: PIL Image tile to rotate
            angle: Rotation angle in degrees
        """
        # Create a larger transparent canvas
        expanded_tile = Image.new('RGBA', (tile.width * 2, tile.height * 2), (0, 0, 0, 0))
        
        # Paste the tile at the center
        expanded_tile.paste(tile, (tile.width // 2, tile.height // 2))
        
        # Rotate the expanded canvas
        rotated_tile = expanded_tile.rotate(angle, expand=False)
        
        # Crop to the original tile size and center
        x = (rotated_tile.width - tile.width) // 2
        y = (rotated_tile.height - tile.height) // 2
        cropped_tile = rotated_tile.crop((x, y, x + tile.width, y + tile.height))
    
        return cropped_tile

    def _process_image_tiles(self, progress):
        """
        Process the image into tiles with the given rotation progress (0 to 1), filling the background
        with the original image to avoid gaps during rotation.
        """
        width, height = self.image.size
        tile_size = width // self.num_divisions
        tiles_x = width // tile_size
        tiles_y = height // tile_size
        
        output = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        for x in range(tiles_x):
            for y in range(tiles_y):
                box = (x * tile_size, y * tile_size, (x + 1) * tile_size, (y + 1) * tile_size)
                tile = self.image.crop(box)
                
                # Create background with original unrotated image
                background = Image.new('RGB',(tile_size,tile_size),(0,0,0,0))
                
                
                # Calculate rotation angle
                angle = progress * 90
                
                # Alternate rotation direction (checkerboard pattern)
                if (x + y) % 2 == 0:
                    rotated_tile = self._rotate_tile(tile, angle)
                else:
                    rotated_tile = self._rotate_tile(tile, -angle)
                
                # Composite the rotated tile over the unrotated background
                combined_tile = Image.alpha_composite(background.convert('RGBA'), rotated_tile)
                
                # Paste the final tile into the output image
                output.paste(combined_tile, box)
        
        return output

    def animate_rotation(self):
        """
        Generate all frames for the animation.
        """
        frames = []
        
        # Generate frames for the rotation
        for i in range(self.frames_per_rotation + 1):
            progress = i / self.frames_per_rotation
            
            # Use sine easing for smoother motion
            eased_progress = math.sin(progress * math.pi / 2)
            
            frame = self._process_image_tiles(eased_progress)
            frames.append(frame)
        
        return frames

    def save_animation(self, output_path, frame_duration=50, pause_duration=2000):
        """
        Save the animation as a GIF with delays at the start and end.
        
        Args:
            output_path: Path where the GIF will be saved
            frame_duration: Duration of each frame in milliseconds
            pause_duration: Duration of the pause at start and end in milliseconds
        """
        frames = self.animate_rotation()
        
        # Calculate how many frames to pause (2-3 seconds)
        pause_frames = pause_duration // frame_duration
        
        # Add pause at the start and end
        start_pause = [frames[0]] * pause_frames
        end_pause = [frames[-1]] * pause_frames
        
        # Combine pauses with animation
        all_frames = start_pause + frames + end_pause
        
        # Save the animation
        all_frames[0].save(
            output_path,
            save_all=True,
            append_images=all_frames[1:],
            duration=frame_duration,
            loop=0,
            optimize=False
        )


def main():
    folder = "an_oil_painting_of_a_donutan_oil_painting_of_a_cup_of_co_s50_cfg7.5_2025-01-06T17-31-09"
    file = "an_oil_painting_of_a_donutan_oil_painting_of_a_cup_of_co_000001.png"
    path =  f"outputs/sd3.5_medium/{folder}/{file}"
    processor = ImageAnimator(path, 3)
    processor.save_animation(f"animations/FLOWERCOFFEE.gif", 50)

if __name__ == "__main__":
    main()