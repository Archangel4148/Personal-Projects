from pathlib import Path
import random
import typing
import pygame

def load_tileset(folder_path: Path, allowed_rotations: list[float]):
    """Load all tile images from the provided folder creating textures for each new rotation (ignoring duplicate rotations)"""
    image_paths = folder_path.glob("*.png")
    tile_map = {}
    edge_checking = set()

    for path in image_paths:
        name = path.stem
        for i, rotation in enumerate(allowed_rotations):
            new_texture = Texture(image_path=path, rotation=rotation)
            edge_tuple = tuple(map(tuple, new_texture.edges.values()))
            if edge_tuple not in edge_checking:
                tile_map[f"{name}_{i+1}"] = new_texture
                edge_checking.add(edge_tuple)
    return tile_map

class Texture:
    def __init__(self, image_path: str = None, rotation: float = 0.0):
        self.image_path = None
        self.rotation = rotation
        self.surface = None
        self.original_surface = None
        self.edges = None
        self.w, self.h = None, None
        self.load_and_process(image_path)

    def set_size(self, w: float, h: float):
        self.w, self.h = w, h
        if self.original_surface:
            # Scale the copy for display
            self.surface = pygame.transform.scale(self.original_surface, (int(self.w), int(self.h)))
            self.surface = pygame.transform.rotate(self.surface, -self.rotation)

    def load_and_process(self, image_path: str):
        # Load the image
        raw_image = pygame.image.load(image_path)
        
        # Extract edges
        # px = pygame.PixelArray(raw_image)
        # raw_edges = {
        #     "top":    tuple(px.transpose()[0]),
        #     "bottom": tuple(px.transpose()[-1]),
        #     "left":   tuple(px[0]),
        #     "right":  tuple(px[-1])
        # }
        # px.close()
        
        w, h = raw_image.get_size()

        def normalize(color):
            return (color.r, color.g, color.b, color.a)

        raw_edges = {
            "top":    tuple(normalize(raw_image.get_at((x, 0)))     for x in range(w)),
            "bottom": tuple(normalize(raw_image.get_at((x, h - 1))) for x in range(w)),
            "left":   tuple(normalize(raw_image.get_at((0, y)))     for y in range(h)),
            "right":  tuple(normalize(raw_image.get_at((w - 1, y))) for y in range(h)),
        }

        # Virtually "rotate" the found pixel values
        r = int(self.rotation) % 360
        # if r == 0:
        #     self.edges = raw_edges
        # elif r == 90 or r == -270:
        #     self.edges = {"top": raw_edges["right"], "bottom": raw_edges["left"], "left": raw_edges["top"], "right": raw_edges["bottom"]}
        # elif r == 180 or r == -180:
        #     self.edges = {"top": raw_edges["bottom"], "bottom": raw_edges["top"], "left": raw_edges["right"], "right": raw_edges["left"]}
        # elif r == 270 or r == -90:
        #     self.edges = {"top": raw_edges["left"], "bottom": raw_edges["right"], "left": raw_edges["bottom"], "right": raw_edges["top"]}

        def reverse(edge):
            return tuple(reversed(edge))

        if r == 0:
            self.edges = raw_edges

        elif r == 90:  # 90° clockwise
            self.edges = {
                "top":    reverse(raw_edges["left"]),
                "right":  raw_edges["top"],
                "bottom": reverse(raw_edges["right"]),
                "left":   raw_edges["bottom"],
            }

        elif r == 180:
            self.edges = {
                "top":    reverse(raw_edges["bottom"]),
                "right":  reverse(raw_edges["left"]),
                "bottom": reverse(raw_edges["top"]),
                "left":   reverse(raw_edges["right"]),
            }

        elif r == 270:  # 270° clockwise (or 90° CCW)
            self.edges = {
                "top":    raw_edges["right"],
                "right":  reverse(raw_edges["bottom"]),
                "bottom": raw_edges["left"],
                "left":   reverse(raw_edges["top"]),
            }

        # Rotate
        self.original_surface = raw_image

    # def get_edges(self) -> dict[str, list[int]]:
    #     """Get the array of pixel values along each edge of the texture"""
    #     px_array = pygame.PixelArray(self.original_surface)
        
    #     left = list(px_array[0])
    #     right = list(px_array[-1])
    #     transposed = px_array.transpose()
    #     top = list(transposed[0])
    #     bottom = list(transposed[-1])
        
    #     return {
    #         "top": top,
    #         "bottom": bottom,
    #         "left": left,
    #         "right": right,
    #     }

class Cell:
    def __init__(self, pos: tuple[float, float], w: float, h: float, options: list[str]):
        self.x, self.y = pos
        self.w = w
        self.h = h
        self.options = options
        self.collapsed = False
        self.texture = None

    def collapse(self, tileset: dict[str, Texture]):
        if not self.options:
            return
        
        # Pick a random option
        choice = random.choice(self.options)
        self.options = [choice]
        self.collapsed = True
        # Update the cell's texture
        self.update_texture(tileset[choice])

    @property
    def entropy(self):
        return len(self.options)

    def update_texture(self, texture: Texture):
        if texture is not None:
            texture.set_size(self.w, self.h)
            self.texture = texture

    def draw(self, screen: pygame.Surface):
        """Draw this cell's texture (if any) to the screen"""
        if self.texture and self.texture.surface:
            screen.blit(self.texture.surface, dest=(self.x, self.y))


class GridManager:
    def __init__(self, cells, dim: tuple[int, int]):
        self.cells = cells
        self.rows, self.cols = dim

    def get_neighbor(self, index: int, direction: str) -> Cell:
        row = index // self.cols
        col = index % self.cols

        if direction == "top" and row > 0:
            return self.cells[index - self.cols]
        if direction == "bottom" and row < self.rows - 1:
            return self.cells[index + self.cols]
        if direction == "left" and col > 0:
            return self.cells[index - 1]
        if direction == "right" and col < self.cols - 1:
            return self.cells[index + 1]
        return None
    
    @classmethod
    def build_grid(cls, c1: tuple[int, int], c2: tuple[int, int], dim: tuple[int, int], tile_names: list[str]) -> typing.Self:
        """Construct a dim[0] by dim[1] grid of cells spanning from corner c1 to c2 (applies default_texture to each Cell)"""
        x1, y1 = c1
        w = abs(c1[0] - c2[0])
        h = abs(c1[1] - c2[1])

        cell_w = w / dim[1]
        cell_h = h / dim[0]

        # Construct the cells
        cells = []
        for i in range(dim[0]):
            for j in range(dim[1]):
                cell_pos = (x1 + cell_w * j, y1 + cell_h * i)
                cells.append(Cell(pos=cell_pos, w=cell_w, h=cell_h, options=list(tile_names)))
        # Build and return the GridManager
        return cls(cells=cells, dim=dim)

class WFCManager:
    def __init__(self, grid_manager: GridManager, tileset: dict[str, Texture]):
        self.grid = grid_manager
        self.tileset = tileset
        self.directions = ["top", "bottom", "left", "right"]
        self.opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}

    def step(self):
        """Perform one step of wave-function collapse"""
        # Find lowest entropy cell(s)
        uncollapsed = [c for c in self.grid.cells if not c.collapsed]
        if not uncollapsed:
            return False # No more cells to collapse
        uncollapsed.sort(key=lambda c: c.entropy)
        min_entropy = uncollapsed[0].entropy
        
        # Pick randomly from minimum entropy cells
        candidates = [c for c in uncollapsed if c.entropy == min_entropy]
        target_cell = random.choice(candidates)
        
        # Collapse
        target_cell.collapse(self.tileset)
        
        # Propagate (update options for all affected cells)
        target_idx = self.grid.cells.index(target_cell)
        self.propagate(target_idx)
        return True
    
    def propagate(self, start_idx: int):
        """Propagate checking/updating available options for each cell, starting from start_idx"""
        stack = [start_idx]
        while stack:
            curr_idx = stack.pop()
            
            for direction in self.directions:
                # Check each neighbor
                neighbor = self.grid.get_neighbor(curr_idx, direction)
                if neighbor and not neighbor.collapsed:
                    # Update neighbor's options
                    pre_count = len(neighbor.options)
                    neighbor.options = self.filter_valid_options(curr_idx, neighbor, direction)
                    
                    # If neighbor's options changed, propagate again from that neighbor
                    if len(neighbor.options) < pre_count:
                        neighbor_idx = self.grid.cells.index(neighbor)
                        if neighbor_idx not in stack:
                            stack.append(neighbor_idx)

    def filter_valid_options(self, current_idx, neighbor, direction):
        current_cell = self.grid.cells[current_idx]
        valid_options = []
        
        # Check every possible tile in the neighbor against every possible tile in current
        for n_opt in neighbor.options:
            n_edges = self.tileset[n_opt].edges
            match_found = False
            
            for c_opt in current_cell.options:
                c_edges = self.tileset[c_opt].edges
                # If the neighbor's option could possibly match any of the current cell's options, keep it
                if c_edges[direction] == n_edges[self.opposite[direction]]:
                    match_found = True
                    break
            
            if match_found:
                valid_options.append(n_opt)
        
        return valid_options

    def reset(self):
        """Restore the grid to its initial state"""
        initial_options = list(self.tileset.keys())
        for cell in self.grid.cells:
            cell.collapsed = False
            cell.texture = None
            cell.options = list(initial_options)