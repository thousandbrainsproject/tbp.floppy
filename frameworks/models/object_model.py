import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
from tbp.monty.frameworks.models.object_model import (
    GraphObjectModel,
    get_correct_k_n,
    remove_close_points,
    GridObjectModel,
    apply_rf_transform_to_points,
    NumpyGraph,
    torch_graph_to_numpy,
    build_point_cloud_graph,
)
from torch_geometric.data import Data
from scipy.spatial import KDTree
import logging
import copy


class FlopCountingGridObjectModel(GridObjectModel):
    """Extension of GridObjectModel that counts FLOPs for KDTree operations."""

    def __init__(self, object_id, max_nodes, max_size, num_voxels_per_dim):
        """Initialize a flop counting grid object model.

        Args:
            object_id: id of the object
            max_nodes: maximum number of nodes in the graph. Will be k in k winner
                voxels with highest observation count.
            max_size: maximum size of the object in meters. Defines size of objects
                that can be represented and how locations are mapped into voxels.
            num_voxels_per_dim: number of voxels per dimension in the models grids.
                Defines the resolution of the model.
        """
        super().__init__(object_id, max_nodes, max_size, num_voxels_per_dim)
        self.flop_counter = None

    def _count_kdtree_construction_flops(self, points):
        """Helper method to count FLOPs for KDTree construction.
        
        Uses formula (5+k)*n*log(n) where k is dimension of points and n is number of points.
        """
        if self.flop_counter is not None:
            num_points = len(points)
            dim = points.shape[1]  # Should be 3 for x,y,z coordinates
            kdtree_construction_flops = int((5 + dim) * num_points * np.log2(num_points))
            self.flop_counter.add_flops(kdtree_construction_flops)

    def _count_kdtree_query_flops(self, search_points, reference_points, k):
        """Helper method to count FLOPs for KDTree query operation.
        
        Counts FLOPs for:
        1. Tree traversal
        2. Distance calculations
        3. Heap operations
        4. Bounding box checks
        """
        if self.flop_counter is not None:
            num_search_points = len(search_points)
            num_reference_points = len(reference_points)
            dim = search_points.shape[1]  # Should be 3 for x,y,z coordinates

            # Tree Traversal FLOPs
            traversal_flops = num_search_points * dim * np.log2(num_reference_points)
            
            # Distance calculation FLOPs
            num_examined_points = int(np.log2(num_reference_points))
            distance_flops = (
                num_search_points * num_examined_points * (3 * dim + dim + 1)
            )  # dim*(3 ops per dim) + dim additions + 1 sqrt
            
            # Heap operations FLOPs
            heap_flops = num_search_points * num_examined_points * np.log2(max(k, 1))
            
            # Bounding box check FLOPs
            bounding_box_flops = num_search_points * num_examined_points * dim
            
            total_flops = int(
                traversal_flops
                + distance_flops
                + heap_flops
                + bounding_box_flops
            )
            self.flop_counter.add_flops(total_flops)

    def find_nearest_neighbors(
        self,
        search_locations,
        num_neighbors,
        return_distance=False,
    ):
        """Find nearest neighbors in graph for list of search locations.

        Note:
            This is currently using kd tree search. In the future we may consider
            doing this directly by indexing the grids. However, an initial
            implementation of this does not seem to be faster than the kd tree search
            (~5-10x slower). However one must consider that search directly in the grid
            would remove the cost of building the tree. TODO: Investigate this further.

        Returns:
            If return_distance is True, return distances. Otherwise, return indices of
            nearest neighbors.
        """
        # Count FLOPs for KDTree query operation
        if self._location_tree is not None:
            self._count_kdtree_query_flops(
                search_locations,
                self._location_tree.data,
                num_neighbors
            )
            
            # We are using the pretrained graphs and location trees for matching
            (distances, nearest_node_ids) = self._location_tree.query(
                search_locations,
                k=num_neighbors,
                p=2,  # eucledian distance
                workers=1,  # using more than 1 worker slows down run on lambda.
            )

            if return_distance:
                return distances
            else:
                return nearest_node_ids

    def set_graph(self, graph):
        """Set self._graph property and convert input graph to right format.
        Also counts FLOPs for KDTree construction."""
        if type(graph) is not NumpyGraph:
            # could also check if is type torch_geometric.data.data.Data
            logging.debug(f"turning graph of type {type(graph)} into numpy graph")
            graph = torch_graph_to_numpy(graph)
        if self.use_original_graph:
            # Just use pretrained graph. Do not use grids to constrain nodes.
            self._graph = graph
            # Count FLOPs before creating KDTree
            self._count_kdtree_construction_flops(graph.pos)
            self._location_tree = KDTree(
                graph.pos,
                leafsize=40,
            )
        else:
            self._initialize_and_fill_grid(
                locations=graph.pos,
                features=graph.x,
                observation_feature_mapping=graph.feature_mapping,
            )
            self._graph = self._build_graph_from_grids()

    def _build_graph_from_grids(self):
        """Build graph from grids by taking the top k voxels with content.
        Also counts FLOPs for KDTree construction.

        Returns:
            Graph with locations and features at the top k voxels with content.
        """
        top_voxel_idxs = self._get_top_k_voxel_indices()

        locations_at_ids = self._location_grid.to_dense()[
            top_voxel_idxs[0], top_voxel_idxs[1], top_voxel_idxs[2]
        ]
        features_at_ids = self._feature_grid.to_dense()[
            top_voxel_idxs[0], top_voxel_idxs[1], top_voxel_idxs[2]
        ]
        graph = build_point_cloud_graph(
            locations=np.array(locations_at_ids),
            features=np.array(features_at_ids),
            feature_mapping=self._current_feature_mapping,
        )
        
        # Count FLOPs before creating KDTree
        self._count_kdtree_construction_flops(graph.pos)
        
        self._location_tree = KDTree(
            graph.pos,
            leafsize=40,
        )
        return graph

    def update_model(
        self,
        locations,
        features,
        location_rel_model,
        object_location_rel_body,
        object_rotation,
    ):
        """Add new locations and features into grids and rebuild graph.
        
        Also counts FLOPs for KDTree construction using formula (5+k)*n*log(n)
        where k is the dimension of points (3 for x,y,z) and n is number of points.
        """
        rf_locations, rf_features = apply_rf_transform_to_points(
            locations=locations,
            features=features,
            location_rel_model=location_rel_model,
            object_location_rel_body=object_location_rel_body,
            object_rotation=object_rotation,
        )
        (
            feature_array,
            observation_feature_mapping,
        ) = self._extract_feature_array(rf_features)
        logging.info(f"adding {locations.shape[0]} observations")
        self._update_grids(
            locations=rf_locations,
            features=feature_array,
            feature_mapping=observation_feature_mapping,
        )
        new_graph = self._build_graph_from_grids()
        assert not np.any(np.isnan(new_graph.x))
        self._graph = new_graph

        # Count FLOPs before creating KDTree
        self._count_kdtree_construction_flops(new_graph.pos)

        # Create the KDTree after counting FLOPs
        self._location_tree = KDTree(
            new_graph.pos,
            leafsize=40,
        )
