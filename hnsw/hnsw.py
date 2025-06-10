import numpy as np
import heapq
import random
import time
import math
from typing import List, Dict, Tuple, Any, Optional, Set, Callable
from collections import defaultdict

class HNSW:
    def __init__(
        self, 
        dim: int, 
        max_elements: int,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
        distance_func: str = "l2"
    ):
        """
        Initialize the HNSW index
        
        Parameters:
        -----------
        dim: int
            Dimensionality of vectors
        max_elements: int
            Maximum number of elements in the index
        ef_construction: int
            Size of the dynamic candidate list during construction
        M: int
            Maximum number of connections per element per layer
        ef_search: int
            Size of the dynamic candidate list during search
        distance_func: str
            Distance function to use ('l2', 'cosine', 'ip' for inner product)
        """
        self.dim = dim
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        self.M_max = M  # Max number of connections for each element per layer
        self.M_max0 = 2 * M  # Max number of connections for each element at layer 0
        self.ef_search = ef_search
        self.ml = -1  # Maximum level
        self.ep = None  # Entry point
        self.num_elements = 0
        
        # Set distance function
        if distance_func == "l2":
            self.distance = self._l2_distance
        elif distance_func == "cosine":
            self.distance = self._cosine_distance
        elif distance_func == "ip":
            self.distance = self._inner_product_distance
        else:
            raise ValueError(f"Unknown distance function: {distance_func}")
        
        # Initialize data structures
        self.data = np.zeros((max_elements, dim), dtype=np.float32)
        self.ids = np.zeros(max_elements, dtype=np.int32)
        self.levels = np.zeros(max_elements, dtype=np.int32)
        
        # Adjacency lists for each layer
        # Format: {level: {node_id: [neighbors]}}
        self.graph = defaultdict(lambda: defaultdict(list))
        
    def _l2_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute L2 distance between vectors"""
        return np.sum((a - b) ** 2)
    
    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance between vectors"""
        return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _inner_product_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute negative inner product (for similarity search)"""
        return -np.dot(a, b)
    
    def _get_random_level(self) -> int:
        """Generate a random level using exponential distribution"""
        return int(-math.log(random.random()) * (1.0 / math.log(self.M)))
    
    def _select_neighbors_heuristic(
        self, 
        candidates: List[Tuple[float, int]], 
        M: int,
        level: int,
        q_vec: np.ndarray
    ) -> List[int]:
        """
        Select best M neighbors among candidates using the heuristic algorithm
        
        Parameters:
        -----------
        candidates: List[Tuple[float, int]]
            List of (distance, node_id) tuples
        M: int
            Maximum number of neighbors to return
        level: int
            Current level
        q_vec: np.ndarray
            Query vector
            
        Returns:
        --------
        List[int]: Selected neighbor IDs
        """
        # Sort candidates by distance
        candidates.sort()
        
        # If we have fewer candidates than M, return all
        if len(candidates) <= M:
            return [c[1] for c in candidates]
        
        # Otherwise, select M best neighbors
        neighbors = []
        for _, e in candidates[:M]:
            neighbors.append(e)
            
        return neighbors
    
    def _search_layer(
        self, 
        q_vec: np.ndarray, 
        ep: int, 
        ef: int, 
        level: int,
        visited: Optional[Set[int]] = None
    ) -> List[Tuple[float, int]]:
        """
        Search for nearest neighbors in a single layer
        
        Parameters:
        -----------
        q_vec: np.ndarray
            Query vector
        ep: int
            Entry point
        ef: int
            Size of the dynamic candidate list
        level: int
            Current level
        visited: Set[int]
            Set of visited nodes
            
        Returns:
        --------
        List[Tuple[float, int]]: List of (distance, node_id) tuples
        """
        if visited is None:
            visited = set()
            
        # Initialize candidates and visited lists
        candidates = []
        visited.add(ep)
        
        # Calculate distance to entry point
        dist = self.distance(q_vec, self.data[ep])
        
        # Initialize candidates and results with the entry point
        candidates = [(dist, ep)]
        results = [(dist, ep)]
        
        # Continue until all candidates are processed
        while candidates:
            # Get the nearest unvisited element
            curr_dist, curr_id = heapq.heappop(candidates)
            
            # If the farthest result is closer than the closest candidate, we're done
            if results[-1][0] < curr_dist:
                break
                
            # Explore neighbors of the current element
            for neighbor_id in self.graph[level][curr_id]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    
                    # Calculate distance to the neighbor
                    neighbor_dist = self.distance(q_vec, self.data[neighbor_id])
                    
                    # If we have fewer than ef results or the neighbor is closer than the farthest result
                    if len(results) < ef or neighbor_dist < results[-1][0]:
                        # Add to candidates
                        heapq.heappush(candidates, (neighbor_dist, neighbor_id))
                        
                        # Add to results
                        results.append((neighbor_dist, neighbor_id))
                        results.sort()
                        
                        # Keep only ef closest results
                        if len(results) > ef:
                            results.pop()
        
        return results
    
    def add(self, vec: np.ndarray, id: int = None) -> int:
        """
        Add a vector to the index
        
        Parameters:
        -----------
        vec: np.ndarray
            Vector to add
        id: int
            Optional ID for the vector (if None, use internal counter)
            
        Returns:
        --------
        int: ID of the added vector
        """
        if self.num_elements >= self.max_elements:
            raise ValueError("Index is full")
            
        # Normalize vector if using cosine distance
        if self.distance == self._cosine_distance:
            vec = vec / np.linalg.norm(vec)
            
        # Assign ID if not provided
        if id is None:
            id = self.num_elements
            
        # Generate random level
        level = self._get_random_level()
        
        # Update maximum level if needed
        if level > self.ml:
            self.ml = level
            
        # Store vector data
        idx = self.num_elements
        self.data[idx] = vec
        self.ids[idx] = id
        self.levels[idx] = level
        
        # If this is the first element, set it as entry point and return
        if self.ep is None:
            self.ep = idx
            self.num_elements += 1
            return id
            
        # Start from the entry point
        curr = self.ep
        
        # For each level from top to the random level of the new element
        for lc in range(self.ml, level, -1):
            # Search for the closest element at the current level
            curr_dist = self.distance(vec, self.data[curr])
            changed = True
            
            # Find a better entry point at the current level
            while changed:
                changed = False
                
                for neighbor_id in self.graph[lc][curr]:
                    neighbor_dist = self.distance(vec, self.data[neighbor_id])
                    
                    if neighbor_dist < curr_dist:
                        curr = neighbor_id
                        curr_dist = neighbor_dist
                        changed = True
        
        # For each level from the random level down to 0
        for lc in range(min(level, self.ml), -1, -1):
            # Search for ef_construction nearest elements at the current level
            neighbors = self._search_layer(vec, curr, self.ef_construction, lc)
            
            # Select M best neighbors
            M_max = self.M_max0 if lc == 0 else self.M_max
            selected = self._select_neighbors_heuristic(neighbors, M_max, lc, vec)
            
            # Connect the new element to selected neighbors
            self.graph[lc][idx] = selected
            
            # Connect selected neighbors to the new element (bidirectional)
            for neighbor_id in selected:
                # Add the new element to the neighbor's connections
                if idx not in self.graph[lc][neighbor_id]:
                    self.graph[lc][neighbor_id].append(idx)
                    
                    # If the neighbor has too many connections, remove some
                    if len(self.graph[lc][neighbor_id]) > M_max:
                        # Calculate distances to all neighbors
                        dists = []
                        for n in self.graph[lc][neighbor_id]:
                            d = self.distance(self.data[neighbor_id], self.data[n])
                            dists.append((d, n))
                            
                        # Keep only M_max closest neighbors
                        selected_neighbors = self._select_neighbors_heuristic(
                            dists, M_max, lc, self.data[neighbor_id]
                        )
                        self.graph[lc][neighbor_id] = selected_neighbors
            
            # Update entry point for the next level
            curr = idx
            
        # If the new element has a higher level than the current entry point,
        # update the entry point
        if level > self.ml:
            self.ep = idx
            
        self.num_elements += 1
        return id
    
    def search(self, query_vec: np.ndarray, k: int = 10) -> List[Tuple[float, int]]:
        """
        Search for k nearest neighbors
        
        Parameters:
        -----------
        query_vec: np.ndarray
            Query vector
        k: int
            Number of nearest neighbors to return
            
        Returns:
        --------
        List[Tuple[float, int]]: List of (distance, node_id) tuples
        """
        if self.ep is None:
            return []
            
        # Normalize query vector if using cosine distance
        if self.distance == self._cosine_distance:
            query_vec = query_vec / np.linalg.norm(query_vec)
            
        # Start from the entry point
        curr = self.ep
        curr_dist = self.distance(query_vec, self.data[curr])
        
        # For each level from top to bottom
        for level in range(self.ml, 0, -1):
            changed = True
            
            # Find a better entry point at the current level
            while changed:
                changed = False
                
                for neighbor_id in self.graph[level][curr]:
                    neighbor_dist = self.distance(query_vec, self.data[neighbor_id])
                    
                    if neighbor_dist < curr_dist:
                        curr = neighbor_id
                        curr_dist = neighbor_dist
                        changed = True
        
        # Search at the bottom layer with increased ef
        neighbors = self._search_layer(query_vec, curr, self.ef_search, 0)
        
        # Return k nearest neighbors
        return neighbors[:k]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index
        
        Returns:
        --------
        Dict[str, Any]: Statistics
        """
        stats = {
            "num_elements": self.num_elements,
            "max_level": self.ml,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "dimension": self.dim,
            "layers": {}
        }
        
        # Calculate statistics for each layer
        for level in range(self.ml + 1):
            level_nodes = len(self.graph[level])
            if level_nodes == 0:
                continue
                
            connections = 0
            for node_id in self.graph[level]:
                connections += len(self.graph[level][node_id])
                
            avg_connections = connections / level_nodes if level_nodes > 0 else 0
            
            stats["layers"][level] = {
                "nodes": level_nodes,
                "total_connections": connections,
                "avg_connections": avg_connections
            }
            
        return stats
    
    def save(self, filename: str) -> None:
        """
        Save the index to a file
        
        Parameters:
        -----------
        filename: str
            File path to save the index
        """
        data_to_save = {
            "dim": self.dim,
            "max_elements": self.max_elements,
            "ef_construction": self.ef_construction,
            "M": self.M,
            "ef_search": self.ef_search,
            "ml": self.ml,
            "ep": self.ep,
            "num_elements": self.num_elements,
            "data": self.data[:self.num_elements].tolist(),
            "ids": self.ids[:self.num_elements].tolist(),
            "levels": self.levels[:self.num_elements].tolist(),
            "graph": dict(self.graph)
        }
        
        np.save(filename, data_to_save, allow_pickle=True)
        
    @classmethod
    def load(cls, filename: str) -> 'HNSW':
        """
        Load an index from a file
        
        Parameters:
        -----------
        filename: str
            File path to load the index from
            
        Returns:
        --------
        HNSW: Loaded index
        """
        data = np.load(filename, allow_pickle=True).item()
        
        # Create a new index
        index = cls(
            dim=data["dim"],
            max_elements=data["max_elements"],
            ef_construction=data["ef_construction"],
            M=data["M"],
            ef_search=data["ef_search"]
        )
        
        # Restore index state
        index.ml = data["ml"]
        index.ep = data["ep"]
        index.num_elements = data["num_elements"]
        index.data[:index.num_elements] = np.array(data["data"])
        index.ids[:index.num_elements] = np.array(data["ids"])
        index.levels[:index.num_elements] = np.array(data["levels"])
        index.graph = defaultdict(lambda: defaultdict(list))
        
        # Restore graph
        for level, nodes in data["graph"].items():
            for node_id, neighbors in nodes.items():
                index.graph[int(level)][int(node_id)] = neighbors
                
        return index 