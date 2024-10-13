import xml.etree.ElementTree as ET
import numpy as np
from enum import Enum
from dataclasses import dataclass
from PIL import Image, ImageDraw
import json


BLACK = 0
BLUE = 1
ORANGE = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
LIGHT_ORANGE = 7
CYAN = 8
RED = 9
BORDER = 10

color_to_hex = {
    0: '#000000',  # black
    1: '#1E93FF',  # blue
    2: '#F93C31',  # orange
    3: '#4FCC30',  # green
    4: '#FFDC00',  # yellow
    5: '#999999',  # grey
    6: '#E53AA3',  # pink
    7: '#FF851B',  # light orange
    8: '#87D8F1',  # cyan
    9: '#921231',  # red
    10: '#555555',  # border
}

class Grid():
    def __init__(self, grid: np.array = None):
        if grid is not None:
            self.grid = grid
            self.h, self.w = grid.shape
        else:
            self.h, self.w = 9, 9
            self.grid = np.array([[BLACK] * self.w for _ in range(self.h)])
    
    def set_color(self, i: int, j: int, color: int) -> None:
        if 0 <= i < self.h and 0 <= j < self.w:
            self.grid[i, j] = color 
    
    def paste(self, img: np.array, x: int, y:int) -> None:
        for i in range(img.h):
            for j in range(img.w):
                self.set_color(y+i, x+j, img.grid[i, j])
    
    def copy(self, x: int, y: int, w: int, h: int) -> 'Grid':
        return Grid(self.grid[y:y+h, x:x+w])
    
    def copy_translate(self, x: int, y: int, w: int, h: int, dx: int, dy: int) -> None:
        obj = self.copy(x, y, w, h)
        self.paste(obj, x+dx, y+dy)
    
    def rotate_90_(self):
        self.grid = np.rot90(self.grid, k=-1)
        self.h, self.w = self.grid.shape
    
    def rotate_90(self, x: int, y: int, w: int, h: int):
        assert x + w <= self.w and y + h <= self.h
        obj = self.copy(x, y, w, h)
        obj.rotate_90_()
        self.paste(obj, x, y)
    
    def flip_(self, axis: int):
        self.grid = np.flip(self.grid, axis=axis)
    
    def flip(self, x: int, y: int, w: int, h: int, axis: int):
        assert x + w <= self.w and y + h <= self.h 
        obj = self.copy(x, y, w, h)
        obj.flip_(axis)
        self.paste(obj, x, y)
    
    def recolor_(self, target_color: int, replacement_color: int):
        for i in range(self.h):
            for j in range(self.w):
                if self.grid[i, j] == target_color:
                    self.grid[i, j] = replacement_color
    
    def flood_(self, x: int, y: int, replacement_color: int):
        if x < 0 or y < 0 or x >= self.w or y >= self.h:
            return
        if self.grid[y, x] == BLACK:
            return
        self.grid[y, x] = replacement_color
        self.flood_(x+1, y, target_color, replacement_color)
        self.flood_(x-1, y, target_color, replacement_color)
        self.flood_(x, y+1, target_color, replacement_color)
        self.flood_(x, y-1, target_color, replacement_color)

    # Import/Export/Visualize functions  
    @staticmethod
    def from_json_file(json_file_path: str) -> 'Grid':
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        examples = data['train']
        input_grids = [(Grid(np.array(example['input'])), Grid(np.array(example['output']))) for example in examples]
        print(data['test'][0]['input'])
        test_grid = Grid(np.array(data['test'][0]['input']))
        
        # Create a new grid to hold all the grids
        combined_height = sum([grid.h for grid, _ in input_grids]) + test_grid.h + 2 * len(input_grids) + 2
        combined_width = 2 * max([grid.w for grid, _ in input_grids] + [test_grid.w]) + 2 * len(input_grids) + 2
        combined_grid = Grid(np.zeros((combined_height, combined_width), dtype=int))

        # Paste all input grids into the combined grid
        current_y = 0
        for input_grid, output_grid in input_grids:
            combined_grid.paste(input_grid, 0, current_y)
            combined_grid.paste(output_grid, input_grid.w + 2, current_y)
            current_y += input_grid.h + 2

        # Paste the test grid into the combined grid
        combined_grid.paste(test_grid, 0, current_y)
        
        return combined_grid
        
        
    def __repr__(self):
        return str(self.grid)
    
    def print(self):
        print(self.grid)
                
    def to_png(self, file_path: str):
        border_size = 1
        cell_size = 13

        # Calculate image dimensions
        img_width = self.w * cell_size + (self.w + 1) * border_size
        img_height = self.h * cell_size + (self.h + 1) * border_size

        img = Image.new('RGB', (img_width, img_height), color_to_hex[BORDER])
        draw = ImageDraw.Draw(img)

        # Draw colored rectangles for each cell
        for i, row in enumerate(self.grid):
            for j, color in enumerate(row):
                x = j * (cell_size + border_size) + border_size
                y = i * (cell_size + border_size) + border_size
                draw.rectangle([x, y, x + cell_size, y + cell_size], fill=color_to_hex[color], outline=color_to_hex[BORDER])

        # Save the image to file
        img.save(f"{file_path}.png")
        print(f"PNG saved as {file_path}.png")
        
    def to_svg(self, file_path: str):
        border_size = 1
        cell_size=13
        
        # Calculate SVG dimensions
        svg_width = self.w * cell_size + (self.w + 1) * border_size
        svg_height = self.h * cell_size + (self.h + 1) * border_size
        
        # Create the SVG root element
        svg = ET.Element('svg', width=str(svg_width), height=str(svg_height), 
                        xmlns="http://www.w3.org/2000/svg")
        
        # Create a background rectangle
        ET.SubElement(svg, 'rect', width="100%", height="100%", fill=color_to_hex[10])
        
        # Create colored rectangles for each cell
        for i, row in enumerate(self.grid):
            for j, color in enumerate(row):
                x = j * (cell_size + border_size) + border_size
                y = i * (cell_size + border_size) + border_size
                ET.SubElement(svg, 'rect', x=str(x), y=str(y), 
                            width=str(cell_size), height=str(cell_size), 
                            fill=color_to_hex[color])
        
        # Create the SVG tree and save to file
        tree = ET.ElementTree(svg)
        tree.write(f"{file_path}.svg", encoding='unicode', xml_declaration=True)
        print(f"SVG saved as {file_path}.svg")


def or(grid1: Grid, grid2: Grid, color: int) -> Grid:
    combined_grid = Grid(np.zeros((max(grid1.h, grid2.h), max(grid1.w, grid2.w)), dtype=int))
    mask1 = grid1.grid != 0
    mask2 = grid2.grid != 0
    combined_grid.grid = np.where(mask1 | mask2, color, 0)
    return combined_grid
    
def and(grid1: Grid, grid2: Grid) -> Grid:
    combined_grid = Grid(np.zeros((max(grid1.h, grid2.h), max(grid1.w, grid2.w)), dtype=int))
    mask1 = grid1.grid != 0
    mask2 = grid2.grid != 0
    combined_grid.grid = np.where(mask1 & mask2, color, 0)
    return combined_grid

def majority(grid1: Grid, grid2: Grid, grid3: Grid, color) -> Grid:
    combined_grid = Grid(np.zeros((max(grid1.h, grid2.h, grid3.h), max(grid1.w, grid2.w, grid3.w)), dtype=int))
    mask1 = grid1.grid != 0
    mask2 = grid2.grid != 0
    mask3 = grid3.grid != 0
    combined_grid.grid = np.where(mask1 + mask2 + mask3 >= 2, color, 0)
    return combined_grid

if __name__ == "__main__":
    arc_example = Grid.from_json_file("data/training/ff805c23.json")
    arc_example.to_png("arc_example")


