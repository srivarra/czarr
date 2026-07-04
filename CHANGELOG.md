# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🚀 Features
- **nvtx:** Payloads, categories, cached domain object ([cbf9c01](https://github.com/srivarra/czarr/commit/cbf9c0124d56791b9403bcd99fce9ce593d6aa67))
- **bench:** Zarr-read gate + cuFile knob sweep (phase-4 step 6) ([2ebc570](https://github.com/srivarra/czarr/commit/2ebc57019ddc83a7c71a6e353716ac1e88c43cb6))
- **array:** CudaZarrArray fast path through czarr.core.Array ([4a51a91](https://github.com/srivarra/czarr/commit/4a51a91d17881ae5f44bb6244fa871556f36326a))
- **core:** Array/AsyncArray — explicit zarrista-shaped read API ([caa0f54](https://github.com/srivarra/czarr/commit/caa0f549ed720c2f851792cc6bbc47fbd8f87996))
- **lowlevel:** Decode + read_array — batched GPU decode and scatter ([29d8655](https://github.com/srivarra/czarr/commit/29d86554520d1dc7969f0df427d7f0c423f221ea))
- **lowlevel:** Io.read — batched cuFile reads + cufile.configure() ([90bd924](https://github.com/srivarra/czarr/commit/90bd924546e6deb6ab6db38efc37943ea3b4b462))
- **lowlevel:** DecodePlan — derive-once metadata + coalesced range planning ([5964f56](https://github.com/srivarra/czarr/commit/5964f5644eaf9c65235d53f588ee91fc5c19877d))
- Reversible configure_gpu via context-manager token ([30d0675](https://github.com/srivarra/czarr/commit/30d067544cd0e41d3f657d5380c463448d39fd7b))
- **bench:** Czarr-bench CLI + harness ([4746b3b](https://github.com/srivarra/czarr/commit/4746b3bb8f5a8874d1aee338a3d5438091b27346))
- **codecs:** GPU blosc decode via nvCOMP native batched API ([394de32](https://github.com/srivarra/czarr/commit/394de32bd4f9ae12bbdecec5e4bd0f93c32d0d4e))
- **core:** CuFileSlabPool — register-once cuFile arena over VMR slabs ([ce13093](https://github.com/srivarra/czarr/commit/ce13093131061d98f8cb09e25f3664ea939d95ca))
- **codecs:** CzarrShardingCodec — coalescing override of v3 sharding ([745e9ad](https://github.com/srivarra/czarr/commit/745e9ad2f0b66d94f9d7d89f1f9506d1f46d9d2c))
- **storage:** Byte-range coalescer for adjacent chunk reads ([084db26](https://github.com/srivarra/czarr/commit/084db267f1d2ac3df5ca559873dccc12f486760b))
- **filters:** Backend-aware FixedScaleOffset with cccl unary_transform ([bbb8755](https://github.com/srivarra/czarr/commit/bbb87559df88a289cdab686a49f0a5cff69bb470))
- **filters:** Backend-aware Shuffle with cupy fallback ([b8e1f04](https://github.com/srivarra/czarr/commit/b8e1f04d17e13d5a613c7d97f64a89da7d87899f))
- **filters:** Backend-aware Delta with cuda.compute backend ([c2ab502](https://github.com/srivarra/czarr/commit/c2ab5022eb11168278edfff0953a278b2229ef49))
- **pipeline:** Default to microbatched read/decode + NVTX visibility ([a565917](https://github.com/srivarra/czarr/commit/a565917b39b20394d33f36bc3ccfa4dea5616a51))
- **codecs:** Wire CudaBytesBytesCodec for CzarrGpuBuffer prototype ([237f243](https://github.com/srivarra/czarr/commit/237f2435c6a41c929025fe4fdbdb2f0c34cb50a3))
- **core:** CzarrGpuBuffer + CzarrGpuNDBuffer skeleton ([ee76b01](https://github.com/srivarra/czarr/commit/ee76b0146728984359e58cf77b4cd43153940da6))
- **codecs+array:** Two-tier LZ4 + cuFile-by-default factories ([5b35764](https://github.com/srivarra/czarr/commit/5b3576495ef759d80840f2d1cec352c210362abb))
- **array:** CudaZarrArray skeleton — wrap, basic-indexing fast path, factories ([164e427](https://github.com/srivarra/czarr/commit/164e427aab1984bb42d1afa880d6c40c1cd1e6fb))
- **storage:** Batched cuFile reads + pipeline override (epic z3hd9ph7 phases 0-2) ([e6b66d6](https://github.com/srivarra/czarr/commit/e6b66d61af1f4dc1df37751ba6861be864017445))
- **bench:** Pipeline_compare + phase6 perf write-up (Phase 6) ([abf22d2](https://github.com/srivarra/czarr/commit/abf22d201f55fc31b310d2a31ca0378363d6df50))
- **crc32c:** Verify zarr-stock Crc32cCodec compatibility (Phase 5) ([feb875d](https://github.com/srivarra/czarr/commit/feb875d0231f0deb91303a5f760ad8bf87de6dcb))
- **sharding:** Verify Zarr v3 sharding routes through CzarrPipeline (Phase 4) ([68924a4](https://github.com/srivarra/czarr/commit/68924a40e3cfb7329e1aa6b7b0c003aadf086e7d))
- **filters:** GPU ArrayArrayCodecs + Shuffle for Zarr v3 (Phase 3) ([f439157](https://github.com/srivarra/czarr/commit/f439157aa914f4778a1df485344612b51369e492))
- **pipeline:** CzarrPipeline + GPU-direct decode path (Phase 2) ([155bbce](https://github.com/srivarra/czarr/commit/155bbce1938959f763f380b2f681726c25b1a6a6))
- **pipeline:** Cuda.core substrate for CzarrPipeline (Phase 1) ([5652ff1](https://github.com/srivarra/czarr/commit/5652ff1220b9313ee40acc5919bc43d91f571948))
- CuFile poll-mode opt-in via configure_gpu() ([8c0163e](https://github.com/srivarra/czarr/commit/8c0163e6863824a3db6e369a2e5d506ab7fae4ba))
- GPU codecs, batched pipeline, cuFile store for Zarr 3 ([77d4ca5](https://github.com/srivarra/czarr/commit/77d4ca5294164aad228de4503be4a948e17950f2))

### 🐛 Fixes
- **ci:** Grant changelog job pull-requests read ([1a9ce09](https://github.com/srivarra/czarr/commit/1a9ce0907c029a062d1377f079a9e755c18f6542))
- **ci:** Environment-independent ty results; skip SARIF on private repo ([6d9d806](https://github.com/srivarra/czarr/commit/6d9d8066721e40a6bd8e23fb2eb9611732ef0233))
- **tests:** Stop littering the repo root with .gpustore_test_ dirs ([2a0c214](https://github.com/srivarra/czarr/commit/2a0c21440fc4f116636ca44906907c83a568b772))
- Hopper blosc decode, CI matrix, stale docs ([31295ee](https://github.com/srivarra/czarr/commit/31295eef21e70e7922fa1e895b11570811478806))
- **filters:** Flip Shuffle default to cupy for Hopper safety ([7061849](https://github.com/srivarra/czarr/commit/70618490c4196a8ded0b108c5afe00ca4106b42f))
- **filters:** BitRound round-half-to-even matches numcodecs ([b6cfb8a](https://github.com/srivarra/czarr/commit/b6cfb8a3e2e1727238bfc6425b61b88b94846148))
- **configure_gpu:** Revert decode_batch_size default to None ([9accaed](https://github.com/srivarra/czarr/commit/9accaed6723acc5f02cf8fa976892b4adf3a9600))
- **filters:** Unpack v3 metadata wrapper in from_dict ([d61c922](https://github.com/srivarra/czarr/commit/d61c92296f05512191a352a421bdeba009c423db))

### ⚡ Performance
- **io:** Persistent cuFile handle cache + process-wide plan cache ([bf75004](https://github.com/srivarra/czarr/commit/bf75004cbbd710592f9e1fe2344f09f3e3a40162))
- **filters:** Fuse BitRound round-to-even into one ElementwiseKernel ([da833de](https://github.com/srivarra/czarr/commit/da833de5e0701c375517915d3aa2571a63ddb568))
- **codecs/lz4:** Drop per-chunk D2H sync + .copy() in _decode_native ([6e6f384](https://github.com/srivarra/czarr/commit/6e6f384227ded937849802afe9374a01320fee33))

### 🔨 Refactor
- **nvtx:** Switch shim to NVIDIA's nvtx package ([ed55ae8](https://github.com/srivarra/czarr/commit/ed55ae88ec06f5a394d67c6a2ddba8baa7a1085b))
- **array:** Flatten czarr/array/ to one module ([b5133a2](https://github.com/srivarra/czarr/commit/b5133a239a9f2780e2d38ac6059371558fa887cf))
- **codecs:** BackendFilter mixin; drop test-only coalesce surface ([de72976](https://github.com/srivarra/czarr/commit/de7297631e22c42c3490a41edb983a17713af8af))
- Adopt typing.override on ABC overrides (PEP 698) ([3081128](https://github.com/srivarra/czarr/commit/308112829de88c9ae19fc90ecb09dcc5a41fe39e))
- Drop `from __future__ import annotations` repo-wide ([1613b1d](https://github.com/srivarra/czarr/commit/1613b1de9c1c7b610eae49a7316e515cdd4d3078))
- Delete CuFileSlabPool, thin CzarrGpuBuffer to plain cupy ([cfea04e](https://github.com/srivarra/czarr/commit/cfea04e2ed60fd6b73d95ba71ab1180f9f421c61))
- Relocate modules to match the dependency graph ([68ce828](https://github.com/srivarra/czarr/commit/68ce8284396cb5ac4724c3e477ab0fc0ff64893d))
- Dead-code sweep (~1k lines, all benched negatives) ([c887301](https://github.com/srivarra/czarr/commit/c887301c31a3840191ef107f9661eea376d80168))
- **pipeline:** Cuda.core alloc, drop dead StreamPool ([3e9b754](https://github.com/srivarra/czarr/commit/3e9b754e8906f5f04d05f4c49edd82ee3e5215ad))
- Drop dead get_backend_overrides + refresh stale docstrings ([3251279](https://github.com/srivarra/czarr/commit/32512797cc1091e727fb1d15de4fe144085fda6c))
- **codecs:** Dedup _import_cccl into _native package init ([34d2219](https://github.com/srivarra/czarr/commit/34d221944cc5fc3bd500c4d20ee7f34bf3bd75f1))
- **codecs:** Drop native LZ4 kernel — wrap nvCOMP only ([f4de208](https://github.com/srivarra/czarr/commit/f4de2082b8002ec44396a17c2907fa4e9f8b29a7))
- **codecs:** Transparent CzarrShardingCodec via registry ([0d8fa3e](https://github.com/srivarra/czarr/commit/0d8fa3ed8be8ac860ee5422ee2df3f70ed18d828))
- **codecs:** Widen CodecBackend to str for filter-tier names ([3448500](https://github.com/srivarra/czarr/commit/344850049e94298652233c6975d651a895944c47))
- Rename Codec to CudaBytesBytesCodec + split codecs tree ([e09fd57](https://github.com/srivarra/czarr/commit/e09fd57898bd183350987be9d641e55a52897f50))
- Harden runtime + drop redundant CzarrCodecPipeline subclass ([892709d](https://github.com/srivarra/czarr/commit/892709da3a7f98b30b9e09dc82bad62ddb8d3a74))
- **storage:** Use Buffer.create_zero_length() for empty key ([fe1f7a3](https://github.com/srivarra/czarr/commit/fe1f7a34f98da7edb9ca57b394047cb75e78f9e1))

### 📝 Documentation
- Include the changelog in the site ([5d779e9](https://github.com/srivarra/czarr/commit/5d779e955494747b6f8e32609860a2c111f76155))
- Pierre color theme; drop the tautological GPU tag ([8c7232b](https://github.com/srivarra/czarr/commit/8c7232b81e2423fbf163fa90bf622f9fce54f430))
- Adopt zensical palette, tags, grids, tabs, footnotes ([41ed51e](https://github.com/srivarra/czarr/commit/41ed51ec64aa31c7486bbaf30e2bfd55e2cfb93d))
- Keep measurements out of usage pages ([ef1adcc](https://github.com/srivarra/czarr/commit/ef1adccdb79feb69e38c77a84725a4ca8e78b998))
- Rewrite in plain register (drop hype, bold lead-ins, duplication) ([ee5670c](https://github.com/srivarra/czarr/commit/ee5670c48fe3f50a5f3bc76b0851aacfe3641509))
- Zensical site + README rewrite for the two-tier API ([f744c31](https://github.com/srivarra/czarr/commit/f744c31c3e12df6fddcc3542f44f18ea35d7a66d))
- **filters:** Document FSO cccl ~ULP-scale rounding drift ([45a1254](https://github.com/srivarra/czarr/commit/45a125459741a70261af8071a1448c2ab5e5a96b))
- **planning:** Cuda-array research synthesis + LZ4 spike ([16b1818](https://github.com/srivarra/czarr/commit/16b1818bf08c3de3a7f1ceccf266d10f7ab7f3f8))
- Buffer-epic handoff context for new worktree ([7aa5584](https://github.com/srivarra/czarr/commit/7aa5584a8195be99686ded875623c277a24ce98a))
- **planning:** Three follow-up epics — kernel cache, GPU buffer, cross-chunk batching ([652b053](https://github.com/srivarra/czarr/commit/652b05331d380688e0231f72ee6c7fd82425cbfb))
- README with quickstart, compat story, recipes ([d7423fa](https://github.com/srivarra/czarr/commit/d7423fa0954f159c3d7f77df64e583c5becb940a))

### 🧪 Testing
- **alloc:** Assert use_rmm_pool routes allocations, not just MR type ([dc388fa](https://github.com/srivarra/czarr/commit/dc388fa233cca83fa2af179aafd64b90d2076d6e))
- Codec, pipeline, storage, allocator coverage ([8a6d8a3](https://github.com/srivarra/czarr/commit/8a6d8a32746a69e19a81bf524d9149736782d4e6))

### 🏗️ CI & build
- **changelog:** Group by conventional commits; enforce with commitizen ([a372b23](https://github.com/srivarra/czarr/commit/a372b23c80e59ad14a60417a41448d018b59bb78))
- Derive the version from git tags (hatch-vcs) ([e1e16ac](https://github.com/srivarra/czarr/commit/e1e16acb18ae03cc9320bbd3d9710c39fbf8b22a))
- **release:** Skip-existing on TestPyPI for idempotent re-runs ([b9aa4c0](https://github.com/srivarra/czarr/commit/b9aa4c0e6ee21323c31650e27577330abad55a14))
- Suppress superfluous-actions for the auto-update PR action ([ee0abdc](https://github.com/srivarra/czarr/commit/ee0abdc4785e5617b07f09178b0d309206474753))
- Full pipeline — tests, lint, security, build, changelog, releases ([3e78080](https://github.com/srivarra/czarr/commit/3e780801e92ce4df88f2498eba6ca93d8188f370))
- **docs:** Build docs on PRs instead of deploying Pages ([3e6cb5f](https://github.com/srivarra/czarr/commit/3e6cb5ff57a9e9b5d1948b10d52cdb9b4d592e20))
- Drop dead deps scipy, zarrs, structlog ([1d583de](https://github.com/srivarra/czarr/commit/1d583de7bd26a4c587687816fc545240613f01d3))
- Sync uv.lock with current pyproject ([51f3f6e](https://github.com/srivarra/czarr/commit/51f3f6ed67c46dd483261738cbce91080727eb58))
- Configure cu12/cu13 optional extras + uv lockfile ([396e59a](https://github.com/srivarra/czarr/commit/396e59aaef9bc3c1bae67c5f44565be74452cb95))

### 🧰 Maintenance
- Stop tracking .planning/ and .dex ([e6b44e3](https://github.com/srivarra/czarr/commit/e6b44e387bf02ef2444fb2a5bfe23b6d4d234b2d))
- Stop tracking bench/ ([630ae3c](https://github.com/srivarra/czarr/commit/630ae3c3d5ca9b9d798b464eaa7fa59fbd052de6))
- Adopt ty via prek, pay down surfaced lint/type debt ([c6deca4](https://github.com/srivarra/czarr/commit/c6deca4bab346edfa14466611eee2b513bbbfc38))
- **spike:** Cuda-compute toolchain validation probe ([267ef6f](https://github.com/srivarra/czarr/commit/267ef6fa385d49476e5502ef296269b56c34b471))
- Drop pyproject-fmt hook, use entry-points tables ([ab94fd4](https://github.com/srivarra/czarr/commit/ab94fd4f9b38945b0c689e46ad54b666f5fc70bf))
- Remove scverse template scaffold ([da49c11](https://github.com/srivarra/czarr/commit/da49c11aec82ac9406a151d73031968771ea3907))

### 🌀 Miscellaneous
- **results:** H100 A/B for the handle/plan caches ([38c781e](https://github.com/srivarra/czarr/commit/38c781e58c8c984419266784e28f6162b7801b31))
- **results:** H100 phase-4 gate — zarr-read A/B, read-path, cufile knobs ([87d3ee0](https://github.com/srivarra/czarr/commit/87d3ee0968ef35fe9f669dc5ec7beaffbba52cf0))
- **overlap:** Compressor-parametric batch_size sweep ([7bc14ef](https://github.com/srivarra/czarr/commit/7bc14efff7bc3d35b2f4a71af0bc5f09d7922086))
- **profile:** Nsys decode profiling — throughput + peak memory ([e0f6a2e](https://github.com/srivarra/czarr/commit/e0f6a2e06bbdb76d62c079f0c17e164847d2d179))
- **buffer:** Pin slab bench to local NVMe + record register-once result ([879e372](https://github.com/srivarra/czarr/commit/879e3723ca3a1cd59e0b84a410a30c1f26519b49))
- **spike:** FSO three-way parity check vs numcodecs oracle ([32579d1](https://github.com/srivarra/czarr/commit/32579d10edcbf05f20b34529b0bdc14b6ee8ff6d))
- **storage:** Fix e2e sharding bench — uncompressed + smaller workload ([a0039e1](https://github.com/srivarra/czarr/commit/a0039e1cbcf8dc69e34669b35d9abc9a8e58c29e))
- **storage:** End-to-end arr[selection] for sharding coalesce ([49381c2](https://github.com/srivarra/czarr/commit/49381c270229ab0c0b449f46bf730cd41f2ae705))
- **cuda_array:** Expand three_way to 5 paths with kvikio splits ([6c23cdf](https://github.com/srivarra/czarr/commit/6c23cdf4ea0fbd687cf79e9d6984571de9b7165e))
- **cuda_array:** 3-way zarr vs torch vs czarr on H100/H200 ([0c1e810](https://github.com/srivarra/czarr/commit/0c1e810a25b5a6ab4981388a3dd126bdb6b40902))
- Czarr-overlap — NVTX wrappers + decode_batch_size knob ([ace2dfc](https://github.com/srivarra/czarr/commit/ace2dfcb067c772f968a348625214906e31e08f7))
- Czarr-v0.1 — CudaZarrArray + two-tier LZ4 + buffer epic ([01a76d4](https://github.com/srivarra/czarr/commit/01a76d45fad13894a7ab643a960dc374f11cbb57))
- **storage:** Drop CzarrGpuBuffer.empty wiring in _gds_get_sync (regression) ([9d1b13b](https://github.com/srivarra/czarr/commit/9d1b13bb15d33321651f9840df8a04d4085cf47a))
- Buffer epic (Phase 0+1+2) into v0.1 for Phase 3 register-once cuFile ([dd7b3c5](https://github.com/srivarra/czarr/commit/dd7b3c576dcc3cf27b65b334f550d4b7d235b9b9))
- **buffer:** Phase 0 spike — alignment probe + revised backing primitive ([6254fe0](https://github.com/srivarra/czarr/commit/6254fe0a59f7d3095fc1610c479efe96c3981158))
- **kernels:** Shell-driven cold-vs-warm + tracked Hopper cuTile bug ([8cbca5b](https://github.com/srivarra/czarr/commit/8cbca5bb3da84e8b6f39c44d57a5e51135525ced))
- **storage:** Probe cuFile batch_io_submit for small-chunk regime ([c5cc001](https://github.com/srivarra/czarr/commit/c5cc0011c98ad04ee2ac2220771f579e1060fca9))
- CzarrPipeline.read_batch override (epic z3hd9ph7 Phase 2) ([6c7d0fc](https://github.com/srivarra/czarr/commit/6c7d0fc2ad6b0c4a6f6403eb3abee33a03b85051))
- Cross-GPU + czarr-vs-CPU+h2d measurements on Bruno H100/H200 ([8a0fada](https://github.com/srivarra/czarr/commit/8a0fada83f84016ea43f5c9a924f44b1fe2b86de))
- CuFile batch_io vs threaded sync probe ([dab3d95](https://github.com/srivarra/czarr/commit/dab3d952f3dcf179ed7c74335abd7d29780443f0))
- GPU vs CPU benchmark suite + SLURM runners ([e10651d](https://github.com/srivarra/czarr/commit/e10651d3b5b878090b2983383596b71c86756e6e))
- Initialize project from cookiecutter-scverse ([427cc6b](https://github.com/srivarra/czarr/commit/427cc6b6dfc33fa1ff6bf1a454be15635f62bf5b))


<!-- generated by git-cliff -->
