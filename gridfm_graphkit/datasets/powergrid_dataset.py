from gridfm_graphkit.datasets.normalizers import Normalizer, BaseMVANormalizer
from gridfm_graphkit.datasets.transforms import (
    AddEdgeWeights,
    AddNormalizedRandomWalkPE,
)

import os.path as osp
import os
import hashlib
import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
from tqdm import tqdm
from typing import Optional, Callable


# --- Parallel preprocessing helpers -------------------------------------------
# Pass 2 builds one graph per scenario (normalize + edge weights + random-walk
# PE) and writes it to disk. These steps are independent across scenarios, so we
# fan them out over a process pool. The random-walk PE depends only on the graph
# topology (edge_index + edge_weight), which repeats across scenarios, so we
# cache it per distinct topology inside each worker.
_W = {}  # worker-process context, set once by _init_preproc_worker
_PE_CACHE = {}  # worker-local: topology hash -> pe tensor


def _resolve_num_workers():
    """CPU budget for preprocessing. Honors $GRIDFM_PREPROC_WORKERS, else the
    SLURM/cgroup-visible cpu count (sched_getaffinity) minus one, min 1."""
    n = int(os.environ.get("GRIDFM_PREPROC_WORKERS", "0"))
    if n > 0:
        return n
    try:
        return max(1, len(os.sched_getaffinity(0)) - 1)  # respects SLURM cpuset
    except AttributeError:  # not Linux (e.g. macOS)
        return max(1, (os.cpu_count() or 1) - 1)


def _build_graph(node6, onehot, edgeGB, edge_idx, scenario_idx, ctx):
    """Build one normalized PyG graph with positional encoding from raw arrays.

    Pure function of its inputs and ``ctx`` (fit normalizers + dims + a pe cache),
    so the serial and pooled paths produce byte-identical output.
    """
    node6_n = ctx["node_normalizer"].transform(torch.tensor(node6, dtype=torch.float))
    x = torch.cat([node6_n, torch.tensor(onehot, dtype=torch.float)], dim=1)
    y = x[:, : ctx["mask_dim"]]

    edge_attr = ctx["edge_normalizer"].transform(torch.tensor(edgeGB, dtype=torch.float))
    edge_index = torch.tensor(edge_idx, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, scenario_id=scenario_idx)
    data = AddEdgeWeights()(data)

    # PE is a function of (edge_index, edge_weight) only -> cache per topology.
    cache = ctx.get("pe_cache")
    key = (
        hashlib.md5(edge_index.numpy().tobytes()).digest()
        + hashlib.md5(data.edge_weight.numpy().round(6).tobytes()).digest()
    )
    pe = cache.get(key) if cache is not None else None
    if pe is None:
        data = AddNormalizedRandomWalkPE(walk_length=ctx["pe_dim"], attr_name="pe")(data)
        if cache is not None:
            cache[key] = data.pe
    else:
        data["pe"] = pe
    return data


def _init_preproc_worker(node_normalizer, edge_normalizer, pe_dim, mask_dim, norm_method, processed_dir):
    torch.set_num_threads(1)  # avoid (num_workers x intra-op threads) oversubscription
    _W.update(
        node_normalizer=node_normalizer,
        edge_normalizer=edge_normalizer,
        pe_dim=pe_dim,
        mask_dim=mask_dim,
        norm_method=norm_method,
        processed_dir=processed_dir,
        pe_cache=_PE_CACHE,
    )
    _PE_CACHE.clear()


def _build_scenario(payload):
    """Pool worker entry point: build + save one scenario, return its id."""
    scenario_idx, node6, onehot, edgeGB, edge_idx = payload
    data = _build_graph(node6, onehot, edgeGB, edge_idx, scenario_idx, _W)
    fname = f"data_{_W['norm_method']}_{_W['mask_dim']}_{_W['pe_dim']}_index_{scenario_idx}.pt"
    torch.save(data, osp.join(_W["processed_dir"], fname))
    return scenario_idx


class GridDatasetDisk(Dataset):
    """
    A PyTorch Geometric `Dataset` for power grid data stored on disk.
    This dataset reads node and edge CSV files, applies normalization,
    and saves each graph separately on disk as a processed file.
    Data is loaded from disk lazily on demand.

    Args:
        root (str): Root directory where the dataset is stored.
        norm_method (str): Identifier for normalization method (e.g., "minmax", "standard").
        node_normalizer (Normalizer): Normalizer used for node features.
        edge_normalizer (Normalizer): Normalizer used for edge features.
        pe_dim (int): Length of the random walk used for positional encoding.
        mask_dim (int, optional): Number of features per-node that could be masked.
        transform (callable, optional): Transformation applied at runtime.
        pre_transform (callable, optional): Transformation applied before saving to disk.
        pre_filter (callable, optional): Filter to determine which graphs to keep.
    """

    def __init__(
        self,
        root: str,
        norm_method: str,
        node_normalizer: Normalizer,
        edge_normalizer: Normalizer,
        pe_dim: int,
        mask_dim: int = 6,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.norm_method = norm_method
        self.node_normalizer = node_normalizer
        self.edge_normalizer = edge_normalizer
        self.pe_dim = pe_dim
        self.mask_dim = mask_dim
        self.length = None

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load normalization stats if available
        node_stats_path = osp.join(
            self.processed_dir,
            f"node_stats_{self.norm_method}.pt",
        )
        edge_stats_path = osp.join(
            self.processed_dir,
            f"edge_stats_{self.norm_method}.pt",
        )
        if osp.exists(node_stats_path) and osp.exists(edge_stats_path):
            self.node_stats = torch.load(node_stats_path, weights_only=False)
            self.edge_stats = torch.load(edge_stats_path, weights_only=False)
            self.node_normalizer.fit_from_dict(self.node_stats)
            self.edge_normalizer.fit_from_dict(self.edge_stats)

    @property
    def raw_file_names(self):
        return ["pf_node.csv", "pf_edge.csv"]

    @property
    def processed_done_file(self):
        return f"processed_{self.norm_method}_{self.mask_dim}_{self.pe_dim}.done"

    @property
    def processed_file_names(self):
        return [self.processed_done_file]

    def download(self):
        pass

    def process(self):
        node_csv = osp.join(self.raw_dir, "pf_node.csv")
        edge_csv = osp.join(self.raw_dir, "pf_edge.csv")

        cols_to_normalize_node = ['Pd','Qd','Pg','Qg','Vm','Va']
        cols_to_normalize_edge = ["G", "B"]

        # ==========================================
        # PASS 1: ITERATIVE NORMALIZATION STATS
        # ==========================================
        print("Pass 1: Iteratively calculating normalizer statistics...")
        
        node_count = 0
        node_min, node_max, node_sum, node_sum_sq = None, None, None, None
        baseMVA_max = 0.0

        for chunk in pd.read_csv(node_csv, usecols=cols_to_normalize_node, chunksize=100_000):
            t = torch.tensor(chunk.values, dtype=torch.float32)
            n = t.shape[0]
            node_count += n
            
            if node_min is None:
                node_min = t.min(dim=0)[0]
                node_max = t.max(dim=0)[0]
                node_sum = t.sum(dim=0)
                node_sum_sq = (t ** 2).sum(dim=0)
            else:
                node_min = torch.min(node_min, t.min(dim=0)[0])
                node_max = torch.max(node_max, t.max(dim=0)[0])
                node_sum += t.sum(dim=0)
                node_sum_sq += (t ** 2).sum(dim=0)
                
            current_baseMVA = t[:, [0, 1, 2, 3]].max().item()
            if current_baseMVA > baseMVA_max:
                baseMVA_max = current_baseMVA

        node_mean = node_sum / node_count
        node_var = (node_sum_sq / node_count) - (node_mean ** 2)
        node_std = torch.sqrt(torch.clamp(node_var, min=0.0))

        node_params = {
            "min_value": node_min, "max_value": node_max, 
            "mean_value": node_mean, "std_value": node_std,
            "baseMVA": baseMVA_max, "baseMVA_orig": getattr(self.node_normalizer, "baseMVA_orig", 100)
        }
        self.node_normalizer.fit_from_dict(node_params)

        edge_count = 0
        edge_min, edge_max, edge_sum, edge_sum_sq = None, None, None, None

        for chunk in pd.read_csv(edge_csv, usecols=cols_to_normalize_edge, chunksize=100_000):
            t = torch.tensor(chunk.values, dtype=torch.float32)
            n = t.shape[0]
            edge_count += n
            
            if edge_min is None:
                edge_min = t.min(dim=0)[0]
                edge_max = t.max(dim=0)[0]
                edge_sum = t.sum(dim=0)
                edge_sum_sq = (t ** 2).sum(dim=0)
            else:
                edge_min = torch.min(edge_min, t.min(dim=0)[0])
                edge_max = torch.max(edge_max, t.max(dim=0)[0])
                edge_sum += t.sum(dim=0)
                edge_sum_sq += (t ** 2).sum(dim=0)

        edge_mean = edge_sum / edge_count if edge_count > 0 else 0
        edge_var = (edge_sum_sq / edge_count) - (edge_mean ** 2) if edge_count > 0 else 0
        edge_std = torch.sqrt(torch.clamp(edge_var, min=0.0)) if edge_count > 0 else 0

        edge_params = {
            "min_value": edge_min, "max_value": edge_max, 
            "mean_value": edge_mean, "std_value": edge_std,
            "baseMVA": baseMVA_max, "baseMVA_orig": getattr(self.edge_normalizer, "baseMVA_orig", 100)
        }
        self.edge_normalizer.fit_from_dict(edge_params)

        self.node_stats = node_params
        self.edge_stats = edge_params
        torch.save(self.node_stats, osp.join(self.processed_dir, f"node_stats_{self.norm_method}.pt"))
        torch.save(self.edge_stats, osp.join(self.processed_dir, f"edge_stats_{self.norm_method}.pt"))

        # ==========================================
        # PASS 2: PARALLEL SCENARIO SAVING
        # ==========================================
        num_workers = _resolve_num_workers()
        print(f"Pass 2: Processing and saving scenarios ({num_workers} worker(s))...")

        payloads = self._iter_scenario_payloads(
            node_csv, edge_csv, cols_to_normalize_node, cols_to_normalize_edge,
        )

        if num_workers <= 1:
            ctx = {
                "node_normalizer": self.node_normalizer,
                "edge_normalizer": self.edge_normalizer,
                "pe_dim": self.pe_dim,
                "mask_dim": self.mask_dim,
                "pe_cache": {},
            }
            for scenario_idx, node6, onehot, edgeGB, edge_idx in tqdm(payloads):
                data = _build_graph(node6, onehot, edgeGB, edge_idx, scenario_idx, ctx)
                torch.save(
                    data,
                    osp.join(self.processed_dir, f"data_{self.norm_method}_{self.mask_dim}_{self.pe_dim}_index_{scenario_idx}.pt"),
                )
        else:
            from multiprocessing import get_context

            mp = get_context("fork") if hasattr(os, "fork") else get_context("spawn")
            with mp.Pool(
                num_workers,
                initializer=_init_preproc_worker,
                initargs=(
                    self.node_normalizer, self.edge_normalizer, self.pe_dim,
                    self.mask_dim, self.norm_method, self.processed_dir,
                ),
            ) as pool:
                for _ in tqdm(pool.imap_unordered(_build_scenario, payloads, chunksize=8)):
                    pass

        with open(osp.join(self.processed_dir, self.processed_done_file), "w") as f:
            f.write("done")

    def _iter_scenario_payloads(self, node_csv, edge_csv, cols_node, cols_edge):
        """Stream complete scenarios from the chunked node/edge CSVs as raw arrays.

        Buffers chunks until at least one scenario is fully read in both files
        (the last scenario in a buffer may be split across the next chunk), then
        yields its raw arrays. Memory stays bounded to a couple of chunks.
        """
        node_iter = pd.read_csv(node_csv, chunksize=100_000)
        edge_iter = pd.read_csv(edge_csv, chunksize=100_000)
        node_buffer = next(node_iter)
        edge_buffer = next(edge_iter)
        node_iter_active = True
        edge_iter_active = True

        while node_iter_active or edge_iter_active or not node_buffer.empty:
            while len(node_buffer["scenario"].unique()) < 2 and node_iter_active:
                try:
                    node_buffer = pd.concat([node_buffer, next(node_iter)], ignore_index=True)
                except StopIteration:
                    node_iter_active = False

            while len(edge_buffer["scenario"].unique()) < 2 and edge_iter_active:
                try:
                    edge_buffer = pd.concat([edge_buffer, next(edge_iter)], ignore_index=True)
                except StopIteration:
                    edge_iter_active = False

            unique_nodes = node_buffer["scenario"].unique()
            unique_edges = edge_buffer["scenario"].unique()

            complete_nodes = set(unique_nodes[:-1]) if (len(unique_nodes) > 1 and node_iter_active) else set(unique_nodes)
            complete_edges = set(unique_edges[:-1]) if (len(unique_edges) > 1 and edge_iter_active) else set(unique_edges)

            complete_scenarios = sorted(complete_nodes.intersection(complete_edges))

            if not complete_scenarios and not node_iter_active and not edge_iter_active:
                complete_scenarios = sorted(set(unique_nodes).intersection(set(unique_edges)))

            # Partition the buffer once, instead of re-scanning it per scenario.
            node_groups = dict(tuple(node_buffer.groupby("scenario")))
            edge_groups = dict(tuple(edge_buffer.groupby("scenario")))
            for scenario_idx in complete_scenarios:
                node_data = node_groups[scenario_idx]
                edge_data = edge_groups[scenario_idx]
                yield (
                    int(scenario_idx),
                    node_data[cols_node].to_numpy(dtype="float32"),
                    node_data[["PQ", "PV", "REF"]].to_numpy(dtype="float32"),
                    edge_data[cols_edge].to_numpy(dtype="float32"),
                    edge_data[["index1", "index2"]].to_numpy().T.astype("int64"),
                )

            node_buffer = node_buffer[~node_buffer["scenario"].isin(complete_scenarios)]
            edge_buffer = edge_buffer[~edge_buffer["scenario"].isin(complete_scenarios)]

    def len(self):
        if self.length is None:
            files = [
                f
                for f in os.listdir(self.processed_dir)
                if f.startswith(
                    f"data_{self.norm_method}_{self.mask_dim}_{self.pe_dim}_index_",
                )
                and f.endswith(".pt")
            ]
            self.length = len(files)
        return self.length

    def get(self, idx):
        file_name = osp.join(
            self.processed_dir,
            f"data_{self.norm_method}_{self.mask_dim}_{self.pe_dim}_index_{idx}.pt",
        )
        if not osp.exists(file_name):
            raise IndexError(f"Data file {file_name} does not exist.")
        data = torch.load(file_name, weights_only=False)
        if self.transform:
            data = self.transform(data)
        return data

    def change_transform(self, new_transform):
        """
        Temporarily switch to a new transform function, used when evaluating different tasks.

        Args:
            new_transform (Callable): The new transform to use.
        """
        self.original_transform = self.transform
        self.transform = new_transform

    def reset_transform(self):
        """
        Reverts the transform to the original one set during initialization, usually called after the evaluation step.
        """
        if self.original_transform is None:
            raise ValueError(
                "The original transform is None or the function change_transform needs to be called before",
            )
        self.transform = self.original_transform
