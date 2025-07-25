import cuda.bindings.driver as cuda
import torch
from triton.testing import do_bench

from typing import Tuple
from functools import partial

import cutlass
from cutlass.cute.runtime import from_dlpack
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils as utils
import cutlass.pipeline as pipeline


class Kernel:
    def __init__(self):
        # Config
        ######################################################
        self.bM = 128
        self.bN = 256
        self.bK = 64

        self.num_consumer = 2
        self.num_producer = 1

        self.atom_layout_mnk = (self.num_consumer, 1, 1)
        self.acc_dtype = cutlass.Float32

        self.op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        self.op_cluster = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        self.stage = 4

        self.shared_storage = None

        self.cluster_shape_mnk = (2, 1, 1)
        ######################################################

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream,
    ):
        # Cluster
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.num_mcast_ctas_a = self.cluster_shape_mnk[1]
        self.num_mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Prepare MMA

        # Make a tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
        # By default, the MMA atom is created with SMEM operand source for A.
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            mA.element_type,
            mB.element_type,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.bN),
        )

        # SMEM Layouts:

        # S<3,4,3> o 0 o (8,64):(64,1) -> Vectorized and swizzled
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                cutlass.utils.layout.LayoutEnum.ROW_MAJOR,
                mA.element_type,
                self.bK,
            ),
            mA.element_type,
        )
        # repeats the SMEM Layout atom to tile the whole tensor shape
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append((self.bM, self.bK), self.stage),
            order=(0, 1, 2),
        )
        # S<3,4,3> o 0 o (8,64):(64,1) -> Vectorized and swizzled
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                cutlass.utils.layout.LayoutEnum.ROW_MAJOR,
                mB.element_type,
                self.bK,
            ),
            mB.element_type,
        )
        # repeats the SMEM Layout atom to tile the whole tensor shape
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append((self.bN, self.bK), self.stage),
            order=(0, 1, 2),
        )

        # TMA

        # A - No need for stages because we will employ TMA for each stage
        # separately
        smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            self.op if self.cluster_shape_mnk[1] == 1 else self.op_cluster,
            mA,
            smem_layout,
            (self.bM, self.bK),
            num_multicast=self.cluster_shape_mnk[1],
        )

        # B - No need for stages because we will employ TMA for each stage
        # separately
        smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            self.op if self.cluster_shape_mnk[0] == 1 else self.op_cluster,
            mB,
            smem_layout,
            (self.bN, self.bK),
            num_multicast=self.cluster_shape_mnk[0],
        )

        M = mA.layout.shape[0]
        N = mB.layout.shape[0]

        # tile_sched_params, grid = self._compute_grid(M, N, self.bM, self.bN)
        tile_sched_params, grid = self._compute_grid(
            M,
            N,
            self.bM,
            self.bN,
            (self.cluster_shape_mnk[0], self.cluster_shape_mnk[1]),
        )

        num_threads = 128 * (self.num_consumer + self.num_producer)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.stage * 2  # Full and Empty Membar
            ]
            sa: cute.struct.Align[
                cute.struct.MemRange[
                    mA.element_type, cute.cosize(a_smem_layout_staged)
                ],
                1024,  # Alignment
            ]
            sb: cute.struct.Align[
                cute.struct.MemRange[
                    mB.element_type, cute.cosize(b_smem_layout_staged)
                ],
                1024,  # Alignment
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            mC,
            tiled_mma,
            a_smem_layout_staged,
            b_smem_layout_staged,
            tile_sched_params,
            self.cta_layout_mnk,
        ).launch(
            grid=grid,
            block=[num_threads, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        mC_mn: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: cutlass.utils.PersistentTileSchedulerParams,
        cta_layout_mnk: cute.Layout,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Get cta/warp/thread idx
        # ///////////////////////////////////////////////////////////////////////////////
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get mcast mask
        # ///////////////////////////////////////////////////////////////////////////////
        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )

        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

        # ///////////////////////////////////////////////////////////////////////////////
        #  Setup TMA Copy Bytes (per Stage)
        # ///////////////////////////////////////////////////////////////////////////////
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            cutlass.BFloat16, a_smem_layout
        ) + cute.size_in_bytes(cutlass.BFloat16, b_smem_layout)

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # mbar arrays
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Threads/warps participating in the pipelines
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        # Each warp will constribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        num_warps = 8
        consumer_arrive_cnt = mcast_size * num_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt, alignment=consumer_arrive_cnt
        )

        # States
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.stage
        )
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.stage
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        cute.arch.cluster_arrive_relaxed()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/B. {a|b}_smem_layout_staged have both
        #  inner transformation swizzle composed with ordinary layout.
        # ///////////////////////////////////////////////////////////////////////////////
        sa = storage.sa.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sb = storage.sb.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        # ///////////////////////////////////////////////////////////////////////////////
        gA_mk = cute.local_tile(mA_mk, (self.bM, self.bK), (None, None))
        gB_nk = cute.local_tile(mB_nk, (self.bN, self.bK), (None, None))
        gC_mn = cute.local_tile(mC_mn, (self.bM, self.bN), (None, None))

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition global tensor for TiledMMA_A/B/C
        # //////////////////////////////////////////////////////////////////////////////
        num_threads_per_warpgroup = 128
        warp_group_idx = cute.arch.make_warp_uniform(tidx // num_threads_per_warpgroup)
        warp_idx_in_warpgroup = cute.arch.warp_idx() % 4
        warp_idx_in_warpgroup = cute.arch.make_warp_uniform(warp_idx_in_warpgroup)

        thr_mma = tiled_mma.get_slice(tidx)

        tCgC = thr_mma.partition_C(gC_mn)

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition shared tensor for TMA load A/B
        # //////////////////////////////////////////////////////////////////////////////
        #  TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        sa_for_tma_partition = cute.group_modes(sa, 0, 2)
        gA_for_tma_partition = cute.group_modes(gA_mk, 0, 2)
        tAsA, tAgA_mk = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            sa_for_tma_partition,
            gA_for_tma_partition,
        )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        sb_for_tma_partition = cute.group_modes(sb, 0, 2)
        gB_for_tma_partition = cute.group_modes(gB_nk, 0, 2)
        tBsB, tBgB_nk = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            sb_for_tma_partition,
            gB_for_tma_partition,
        )

        # //////////////////////////////////////////////////////////////////////////////
        #  Make frangments
        # //////////////////////////////////////////////////////////////////////////////
        tCsA = thr_mma.partition_A(sa)
        tCsB = thr_mma.partition_B(sb)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        cute.arch.cluster_wait()

        k_tile_cnt = cute.size(tAgA_mk, mode=[2])
        # Producer
        if warp_group_idx < 1:
            cute.arch.warpgroup_reg_dealloc(24)
            if warp_idx_in_warpgroup == 0:
                tile_sched = cutlass.utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )

                work_tile = tile_sched.initial_work_tile_info()

                while work_tile.is_valid_tile:
                    tile_m, tile_n, _ = work_tile.tile_idx

                    for tile_k in cutlass.range(k_tile_cnt):
                        mainloop_pipeline.producer_acquire(mainloop_producer_state)

                        tAgA_k_index = (None, tile_m, tile_k)

                        tAsA_stage_index = (
                            None,
                            mainloop_producer_state.index,
                        )
                        tBgB_k_index = (None, tile_n, tile_k)
                        tBsB_stage_index = (None, mainloop_producer_state.index)

                        tAgA_k = tAgA_mk[tAgA_k_index]
                        tAsA_pipe = tAsA[tAsA_stage_index]
                        tBgB_k = tBgB_nk[tBgB_k_index]
                        tBsB_pipe = tBsB[tBsB_stage_index]

                        cute.copy(
                            tma_atom_a,
                            tAgA_k,
                            tAsA_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                            mcast_mask=a_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_k,
                            tBsB_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                            mcast_mask=b_mcast_mask,
                        )

                        mainloop_pipeline.producer_commit(mainloop_producer_state)
                        mainloop_producer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

        else:
            cute.arch.warpgroup_reg_alloc(240)

            tile_sched = cutlass.utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )

            work_tile = tile_sched.initial_work_tile_info()
            num_k_blocks = cute.size(tCrA, mode=[2])

            accumulators = cute.make_fragment(
                tCgC[None, None, None, 0, 0].shape, self.acc_dtype
            )
            while work_tile.is_valid_tile:
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

                tile_m, tile_n, _ = work_tile.tile_idx

                for k_tile in cutlass.range(k_tile_cnt):
                    # /////////////////////////////////////////////////////////////////////////////
                    #  Wait for TMA copies to complete
                    # /////////////////////////////////////////////////////////////////////////////
                    mainloop_pipeline.consumer_wait(mainloop_consumer_state)
                    # /////////////////////////////////////////////////////////////////////////////
                    #  WGMMA
                    # /////////////////////////////////////////////////////////////////////////////
                    cute.nvgpu.warpgroup.fence()
                    for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_state.index,
                        )
                        tCrA_1phase = tCrA[k_block_coord]
                        tCrB_1phase = tCrB[k_block_coord]

                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA_1phase,
                            tCrB_1phase,
                            accumulators,
                        )
                        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

                    cute.nvgpu.warpgroup.commit_group()
                    # Wait on the wgmma barrier for previous k_pipe_mmas wgmmas to complete
                    cute.nvgpu.warpgroup.wait_group(0)

                    mainloop_pipeline.consumer_release(mainloop_consumer_state)
                    mainloop_consumer_state.advance()

                store_copy = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), self.acc_dtype
                )
                cute.copy(
                    store_copy, accumulators, tCgC[None, None, None, tile_m, tile_n]
                )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        return

    @staticmethod
    def _compute_grid(
        M: int,
        N: int,
        bM: int,
        bN: int,
        cluster_shape_mn: Tuple[int, int],
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        num_ctas_mnl = (M // bM, N // bN, 1)
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        max_active_clusters = cutlass.const_expr(128)  # Hardware

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid


def run():
    M, N, K = 8192, 8192, 8192
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    a_tensor = from_dlpack(a, assumed_align=16)  # (M, K) : (K, 1) - K-Major
    b_tensor = from_dlpack(b, assumed_align=16)  # (N, K) : (K, 1) - K-Major
    c_tensor = from_dlpack(c, assumed_align=16)  # (M, N) : (N, 1) - N-Major

    kernel = Kernel()

    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    wgmma = cute.compile(kernel, a_tensor, b_tensor, c_tensor, stream=current_stream)

    wgmma(a_tensor, b_tensor, c_tensor, current_stream)
    torch.cuda.synchronize()
    c_ref = torch.matmul(a, b.t())
    torch.testing.assert_close(c.to(torch.bfloat16), c_ref, atol=1e-03, rtol=1e-03)

    wgmma_callable = partial(wgmma, a_tensor, b_tensor, c_tensor, current_stream)

    avg_time = do_bench(wgmma_callable, warmup=500, rep=10000)
    print(f"Time in ms = {avg_time}")
    print(f"TFLOPs = {(2 * M * N * K / 1e12) / (avg_time / 1000)}")


if __name__ == "__main__":
    run()
