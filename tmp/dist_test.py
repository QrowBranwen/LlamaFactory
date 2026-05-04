import os
import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")

x = torch.ones(1, device=f"cuda:{local_rank}") * rank
dist.all_reduce(x)

if rank == 0:
    print("all_reduce result:", x.item())
    print("expected:", sum(range(world_size)))

dist.barrier()
print(f"rank {rank}/{world_size} local_rank {local_rank} ok", flush=True)

dist.destroy_process_group()