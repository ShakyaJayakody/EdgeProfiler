from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class ModelConfig:
    name: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_heads: int = 32
    vocab_size: int = 50257
    seq_len: int = 1024

@dataclass
class HardwareConfig:
    name: str
    peak_flops: float             # in FLOPs/sec
    memory_bandwidth: float       # in bytes/sec
    storage_bandwidth: float      # bytes/sec for model load
    host_to_device_bw: float      # bytes/sec for PCIe
    network_bandwidth: float      # bytes/sec for distributed shards
    compute_util: float = 0.7     # effective utilization of peak FLOPs
    memory_util: float = 0.6      # effective utilization of memory BW
    storage_util: float = 0.8     # effective util for storage BW
    h2d_util: float = 0.7         # effective util for host→device BW
    net_util: float = 0.8         # effective util for network BW
    energy_per_flop: float = 1e-12    # Joules per FLOP
    energy_per_byte: float = 1e-9     # Joules per accessed byte

@dataclass
class PrecisionConfig:
    name: str
    dtype_size: int               # bytes per element

class EdgeProfile:
    def __init__(self, model: ModelConfig, hw: HardwareConfig, prec: PrecisionConfig):
        self.model = model
        self.hw = hw
        self.prec = prec

    def parameter_count(self) -> int:
        # Include weights for projections and FFN and embeddings
        L, H, I = self.model.num_layers, self.model.hidden_size, self.model.intermediate_size
        # Q,K,V,O: 4 projections HxH per layer
        proj = L * 4 * H * H
        # FFN: HxI + I x H per layer
        ffn  = L * (H * I + I * H)
        # embeddings + output head
        embed = 2 * self.model.vocab_size * H
        return proj + ffn + embed

    def flops_per_token(self) -> float:
        L, H, I, S = self.model.num_layers, self.model.hidden_size, self.model.intermediate_size, self.model.seq_len
        attn_proj = L * (3 * 2 * H * H + 2 * H * H)
        attn_kv   = L * (2 * H * S + 2 * H * S)
        ffn       = L * (2 * H * I + 2 * I * H)
        ln        = L * (2 * H + 2 * H)
        softmax   = L * (5 * H)
        return attn_proj + attn_kv + ffn + ln + softmax

    def memory_footprint(self) -> float:
        # Weights + activations + KV cache
        params = self.parameter_count()
        weight_bytes = params * self.prec.dtype_size
         # Activations: hidden states for full sequence
        act_bytes    = self.model.seq_len * self.model.hidden_size * self.prec.dtype_size
        # KV cache: stored keys & values per layer
        kv_bytes     = self.model.num_layers * self.model.seq_len * self.model.hidden_size * 2 * self.prec.dtype_size
        return weight_bytes + act_bytes + kv_bytes

    def breakdown_times(self) -> Dict[str, float]:
        L, H, I, S = self.model.num_layers, self.model.hidden_size, self.model.intermediate_size, self.model.seq_len
        # raw FLOPs
        attn_proj = L * (3 * 2 * H * H + 2 * H * H)
        attn_kv   = L * (2 * H * S + 2 * H * S)
        ffn       = L * (2 * H * I + 2 * I * H)
        ln        = L * (2 * H + 2 * H)
        softmax   = L * (5 * H)
        # times in seconds
        comp_util = self.hw.peak_flops * self.hw.compute_util
        return {
            'attn_proj_ms': (attn_proj / comp_util) * 1e3,
            'attn_kv_ms':   (attn_kv   / comp_util) * 1e3,
            'ffn_ms':       (ffn       / comp_util) * 1e3,
            'ln_ms':        (ln        / comp_util) * 1e3,
            'softmax_ms':   (softmax   / comp_util) * 1e3,
        }

    def estimate_latency(self) -> Tuple[float, float, float]:
        flops = self.flops_per_token()
        weight_bytes = self.parameter_count() * self.prec.dtype_size
        comp_time = flops / (self.hw.peak_flops * self.hw.compute_util)
        mem_time  = self.memory_footprint() / (self.hw.memory_bandwidth * self.hw.memory_util)
        return comp_time, mem_time, max(comp_time, mem_time)

    def io_time(self) -> float:
        weight_bytes = self.parameter_count() * self.prec.dtype_size
        return weight_bytes / (self.hw.storage_bandwidth * self.hw.storage_util)

    def h2d_transfer_time(self) -> float:
        weight_bytes = self.parameter_count() * self.prec.dtype_size
        return weight_bytes / (self.hw.host_to_device_bw * self.hw.h2d_util)

    def network_transfer_time(self) -> float:
        
        shard_bytes = self.model.seq_len * self.model.hidden_size * self.prec.dtype_size
        return shard_bytes / (self.hw.network_bandwidth * self.hw.net_util)

    def arithmetic_intensity(self) -> float:
        return self.flops_per_token() / self.memory_footprint()

    def energy_estimate(self) -> float:
        e_flop = self.flops_per_token() * self.hw.energy_per_flop
        e_mem  = self.memory_footprint() * self.hw.energy_per_byte
        return e_flop + e_mem

    def full_profile(self) -> dict:
        comp, mem, base = self.estimate_latency()
        io    = self.io_time()
        h2d   = self.h2d_transfer_time()
        net   = self.network_transfer_time()
        end2end = base + io + h2d + net
        br = self.breakdown_times()

        return {
            'device': self.hw.name,
            'model': self.model.name,
            'precision': self.prec.name,
            'params (M)': self.parameter_count() / 1e6,
            'FLOPs/token (G)': self.flops_per_token() / 1e9,
            'mem_footprint (MB)': self.memory_footprint() / (1024**2),
            'compute_ms': comp  * 1e3,
            'memory_ms':  mem  * 1e3,
            'io_ms':      io   * 1e3,
            'h2d_ms':     h2d  * 1e3,
            'net_ms':     net  * 1e3,
            'end2end_ms': end2end * 1e3,
            'arithmetic_intensity': self.arithmetic_intensity(),
            'energy_J': self.energy_estimate(),
            **br
        }

if __name__ == '__main__':
    # Precisions
    precisions = [
        PrecisionConfig('FP32', 4),
        PrecisionConfig('FP16', 2),
        PrecisionConfig('INT8', 1),
    ]

    # Models
    models = [
        ModelConfig('TinyLlama-1B',    24,2048,8192, num_heads=32, seq_len=1024),
        ModelConfig('Gemma3-1B',       24,2048,8192, num_heads=32, seq_len=1024),
        ModelConfig('Llama3.2-1B',     24,2048,8192, num_heads=32, seq_len=1024),
        ModelConfig('DeepSeek-r1-1.5B',26,2304,9216, num_heads=36, seq_len=1024),
        # ModelConfig('2B', 28, 2560, 10240, num_heads=40, seq_len=1024),
        # ModelConfig('2.5B', 30, 2816,11264, num_heads=44, seq_len=1024),
        # ModelConfig('3B', 32, 3072,12288, num_heads=48, seq_len=1024),
    ]
    # Hardware platforms
    hardware_list = [
        HardwareConfig('Raspberry Pi 4',   50e9, 19e9, 5e8, 12e9, 1e8),
        HardwareConfig('Raspberry Pi 5',   64e9, 34e9, 5e8, 16e9, 1e8),
        HardwareConfig('Jetson Nano Super',0.5e12,25.6e9, 2e9, 30e9, 5e8),
    ]
    # Header
    header = (
        f"{'Device':>17} | {'Model':>15} | {'Prec':>5} | {'Params(M)':>9} |"
        f" {'FLOPs/tok(G)':>12} | {'Mem(MB)':>8} | {'Comp(ms)':>8} | {'Mem(ms)':>8} |"
        f" {'IO(ms)':>7} | {'H2D(ms)':>7} | {'Net(ms)':>7} | {'E2E(ms)':>8} |"
        f" {'AI':>6} | {'Energy(J)':>10}"
    )
    print(header)
    print('-' * len(header))

    for hw in hardware_list:
        for model in models:
            for prec in precisions:
                prof = EdgeProfile(model, hw, prec).full_profile()
                print(
                    f"{prof['device']:>17} | {prof['model']:>15} | {prof['precision']:>5} | "
                    f"{prof['params (M)']:9.1f} | {prof['FLOPs/token (G)']:12.1f} | "
                    f"{prof['mem_footprint (MB)']:8.1f} | {prof['compute_ms']:8.2f} | "
                    f"{prof['memory_ms']:8.2f} | {prof['io_ms']:7.2f} | {prof['h2d_ms']:7.2f} | "
                    f"{prof['net_ms']:7.2f} | {prof['end2end_ms']:8.2f} | "
                    f"{prof['arithmetic_intensity']:6.2f} | {prof['energy_J']:10.3f}"
                )
