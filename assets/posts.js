// Every post on the site, with its category and tags. The blog page
// (/blog/) renders its timeline / category / series / tag views from this
// list, and post pages use it to show their tags and related posts.
//
// To add a post: add an entry at the TOP (newest first). kind is
// article | interactive; cat is a key of CATEGORIES; reuse existing
// tags where possible. Then run `python3 scripts/post-links.py` to refresh
// the cross-reference index in assets/post-links.js.
(() => {
  const CATEGORIES = {
    pretraining: '预训练算法',
    training: '训练系统',
    inference: '推理系统',
    kernels: 'GPU 与算子',
    models: '模型架构与报告',
    recipes: '训练方法',
    basics: '入门与随笔',
  };
  const KINDS = { article: ['文章', 'kind'], interactive: ['交互', 'kind interactive'] };

  const POSTS = [
    {"slug": "moe-upcycling", "title": "已有模型怎样长成 MoE：从权重复用到专家分化", "date": "2026-10-22", "kind": "interactive", "cat": "pretraining", "tags": ["预训练", "MoE", "Upcycling", "实验"], "desc": "从复制 dense FFN 的函数保持条件，推导 router 梯度、SwiGLU 细粒度切分与专家分化，再核算 upcycling 的训练预算、GEMM 和通信。含交互图、四道习题和 NumPy 实验。"},
    {"slug": "sparse-pretraining", "title": "稀疏预训练：动态连接怎样变成真实加速", "date": "2026-10-22", "kind": "interactive", "cat": "pretraining", "tags": ["预训练", "稀疏", "Tensor Core", "优化"], "desc": "串起RigL、SR-STE与2:4 Transformer预训练，推导掩码更新、可转置约束、三个GEMM和优化器账本，配可操作数值图与实验。"},
    {"slug": "recurrent-depth", "title": "同一组参数多算几遍：共享深度与递归推理", "date": "2026-10-22", "kind": "interactive", "cat": "models", "tags": ["架构", "递归", "深度", "KV cache"], "desc": "从Universal Transformer和ALBERT到Recurrent Depth与MoR，推导共享参数的梯度、截断反传、动态轮数与每轮KV缓存协议，结合数值图与NumPy实验。"},
    {"slug": "state-space-models", "title": "状态空间模型怎样记住历史：从 S4 到 Mamba-3", "date": "2026-10-22", "kind": "interactive", "cat": "models", "tags": ["架构", "状态空间", "Mamba", "递归"], "desc": "从连续状态方程到递推、卷积和SSD，推导S4、Mamba、Mamba-2与Mamba-3的离散化、状态旋转和MIMO，并验证精度、分块与状态生命周期。"},
    {"slug": "weight-pruning", "title": "权重剪枝：从幅值到数据感知重建", "date": "2026-10-22", "kind": "interactive", "cat": "inference", "tags": ["稀疏", "剪枝", "推理", "压缩"], "desc": "从输出重建推导幅值剪枝、Wanda、SparseGPT与MaskLLM，连接非结构和2:4格式的真实成本，含交互计算、习题与NumPy实验。"},
    {"slug": "activation-sparsity", "title": "激活稀疏：当前 token 需要哪些神经元", "date": "2026-10-22", "kind": "interactive", "cat": "inference", "tags": ["稀疏", "激活", "推理", "访存"], "desc": "从ReLU、Deja Vu到TEAL与2:4激活稀疏，推导截断误差、batch访问并集和前后向模式，解释实际GPU成本与诊断实验。"},
    {"slug": "model-merging", "title": "多个 checkpoint 怎样合成一个模型：从 Model Soups 到参数对齐", "date": "2026-10-22", "kind": "interactive", "cat": "recipes", "tags": ["模型合并", "微调", "优化", "实验"], "desc": "把模型合并拆成坐标对齐、任务更新和验证三个问题：推导 Model Soups、Task Arithmetic、TIES 与 DARE，用交互图和 NumPy 实验检查置换对称、符号冲突、稀疏化方差和词表对齐。"},
    {"slug": "effective-depth", "title": "一个模型到底用了多少深度：测量、剪层与提前退出", "date": "2026-10-22", "kind": "interactive", "cat": "models", "tags": ["架构", "深度", "残差", "提前退出"], "desc": "从ShortGPT、层剪枝与LayerSkip到几何有效深度，推导CKA、残差累加参考值与任务干预，学习设计剪层、缓存和早退的受控实验。"},
    {"slug": "over-encoding", "title": "Over-Encoding：用查表扩展语言模型容量", "date": "2026-10-22", "kind": "interactive", "cat": "models", "tags": ["架构", "Embedding", "条件记忆", "分片"], "desc": "从输入输出成本不对称推导 n-gram 哈希、低维切片与碰撞梯度，比较 LookupLM、N-Grammer、OE 和行分片通信，附交互图与 NumPy 实验。"},
    {"slug": "engram", "title": "Engram：条件记忆怎样与 MoE 分工", "date": "2026-10-22", "kind": "interactive", "cat": "models", "tags": ["架构", "Engram", "条件记忆", "访存"], "desc": "从确定性 n-gram 检索推导上下文门控、短卷积与流式状态，再计算记忆预算和 host 预取窗口，附受控实验、习题与 NumPy 实验。"},
    {"slug": "concept-models", "title": "从 token 到概念：句向量生成与动态分段", "date": "2026-10-22", "kind": "interactive", "cat": "models", "tags": ["架构", "概念模型", "压缩", "学习目标"], "desc": "从 Meta LCM 的句向量分布生成到 DLCM 的动态分段，推导条件均值、扩散、因果可见性、KV 副本计权与完整计算成本；含交互图、习题与 NumPy 实验。"},
    {"slug": "pretraining-scaling", "title": "训练预算怎样分配：Scaling Laws 与模型、数据的取舍", "date": "2026-10-22", "kind": "interactive", "cat": "pretraining", "tags": ["预训练", "Scaling Laws", "实验", "数据"], "desc": "从固定 FLOPs 的最优模型规模推导，到跨尺度拟合、重复数据与部署成本：用三个交互图、四道计算与诊断题和 NumPy 实验，学会制定预训练预算。"},
    {"slug": "pretraining-data-curation", "title": "预训练数据怎样筛选：去重、质量与多样性", "date": "2026-10-22", "kind": "interactive", "cat": "pretraining", "tags": ["预训练", "数据", "去重", "实验"], "desc": "从 MinHash 与 SemDeDup 推导去重，再用 FineWeb、DCLM 理解质量过滤、快照分布和固定预算评估，附交互图、计算题与 CPU 实验。"},
    {"slug": "pretraining-data-mixture", "title": "数据配比怎样学出来：DoReMi、DoGE 与 RegMix", "date": "2026-10-22", "kind": "interactive", "cat": "pretraining", "tags": ["预训练", "数据", "优化", "实验"], "desc": "推导参考相对 excess loss、梯度对齐与小模型回归，分析 token 归约、梯度累积和跨规模迁移，附交互图与 NumPy 实验。"},
    {"slug": "pretraining-mup", "title": "从小模型调参到大模型：μP 与超参数迁移", "date": "2026-10-22", "kind": "interactive", "cat": "pretraining", "tags": ["预训练", "μP", "优化器", "超参数"], "desc": "从特征更新推导μP的初始化与SGD、Adam学习率缩放，串起Tensor Programs和后续参数化研究，用交互计算、工程诊断及NumPy实验学习宽度迁移。"},
    {"slug": "pretraining-objectives", "title": "一段文本提供什么监督：从去噪、FIM 到多 token 预测", "date": "2026-10-22", "kind": "interactive", "cat": "pretraining", "tags": ["预训练", "学习目标", "FIM", "MTP"], "desc": "把 T5、UL2、FIM 与多 token 预测放进同一套条件概率、标签和计算预算中，推导监督数量、梯度累积归一化与词表显存，并用 NumPy 检查目标是否写对。"},
    {"slug": "pretraining-stability", "title": "深层 Transformer 怎样稳定训练", "date": "2026-10-22", "kind": "interactive", "cat": "pretraining", "tags": ["预训练", "稳定性", "归一化", "优化器"], "desc": "从残差 Jacobian、Pre-LN 与 DeepNorm 推导深度稳定性，再用注意力熵理解 QK normalization 和 σReparam，建立包含梯度裁剪、参数更新和低精度计算的诊断实验。"},
    {"slug": "hyper-connections", "title": "Hyper-Connections：从一条残差流到多路信息交换", "date": "2026-10-22", "kind": "interactive", "cat": "models", "tags": ["架构", "Hyper-Connections", "残差", "稳定性"], "desc": "从恒等残差出发推导 HC 的读取、混合、写回、动态参数化与梯度；分析信息路径、初始化、带宽成本，含交互图、习题与 NumPy 实验。"},
    {"slug": "manifold-hyper-connections", "title": "mHC：用双随机约束稳定多路残差", "date": "2026-10-22", "kind": "interactive", "cat": "models", "tags": ["架构", "Hyper-Connections", "mHC", "稳定性"], "desc": "推导 mHC 的双随机约束、均值守恒与非扩张性，分析有限 Sinkhorn、动态梯度、FP32 归一化、融合与重算，含交互图、完整习题及 NumPy 实验。"},
    {"slug": "attention-residuals", "title": "Attention Residuals：沿深度选择信息", "date": "2026-10-22", "kind": "interactive", "cat": "models", "tags": ["架构", "注意力", "残差", "显存"], "desc": "从残差累加推导沿深度的 attention，理解 Full 与 Block AttnRes 的权重、梯度和状态生命周期，再用在线 softmax、重算与流水线缓存解释它怎样高效运行。"},
    {"slug": "llm-engineering", "title": "LLM 工程学习路径：从张量到推理服务", "kind": "interactive", "cat": "basics", "tags": ["学习路径", "LLM", "工程", "实验"], "date": "2026-10-20", "desc": "十个阶段串起模型计算、推理服务、预训练、稀疏与低精度训练、模型复用、递归结构和条件记忆；每阶段都有先修、验收任务与实验，最后完成最小推理引擎。"},
    {"slug": "mini-inference-engine", "title": "从零写一个最小推理引擎", "kind": "interactive", "cat": "inference", "tags": ["推理", "KV cache", "调度", "采样", "实验"], "date": "2026-10-20", "desc": "用可运行 NumPy Transformer 连起 GQA、RoPE、分页 KV、逐轮调度与采样，从独立 dense 参考、容量承诺和故障注入建立推理引擎的正确性。"},
    {"slug": "tensor-layouts", "title": "张量布局、stride 与 GPU 访存", "date": "2026-10-20", "kind": "interactive", "cat": "kernels", "tags": ["GPU", "算子优化", "张量布局", "访存"], "desc": "从索引与 stride 推导真实地址，解释转置、切片、复制与别名，再用 sector 和 bank 分析 GPU 访存；含交互图、工程排错、四道习题和 NumPy 实验。"},
    {"slug": "gemm-optimization", "title": "GEMM：从三重循环到 Tensor Core", "date": "2026-10-20", "kind": "interactive", "cat": "kernels", "tags": ["GPU", "算子优化", "GEMM", "Triton"], "desc": "从三重循环推导分块复用、shared memory 与寄存器预算，解释 Tensor Core、尾块和调度，再用实测证据定位性能问题；含动态图、计算题与 NumPy/Triton 实验。"},
    {"slug": "cuda-streams", "title": "CUDA streams、events 与异步流水", "date": "2026-10-20", "kind": "interactive", "cat": "kernels", "tags": ["CUDA", "推理", "性能分析", "并发"], "desc": "从命令队列与依赖边推导异步流水，算清 event、pinned memory、跨流寿命和缓冲复用；含时间线、工程故障分析、四道习题及 CPU/CUDA 实验。"},
    {"slug": "torch-compile", "title": "torch.compile：从 Python 到 GPU kernel", "date": "2026-10-20", "kind": "interactive", "cat": "kernels", "tags": ["PyTorch", "编译", "推理", "算子优化"], "desc": "连接融合、图捕获、guards、动态形状与分桶成本，结合实际编译日志定位重编译和数据依赖问题；含动态图、四道习题与 NumPy/PyTorch 实验。"},
    {"slug": "rope-long-context", "title": "RoPE 与长上下文扩展", "date": "2026-10-20", "kind": "interactive", "cat": "models", "tags": ["RoPE", "长上下文", "位置编码", "KV cache"], "desc": "从二维旋转推导相对位置，连接频率、长上下文缩放与 KV 生命周期；含可操作图解、精度与分块排错、习题和 NumPy 实验。"},
    {"slug": "sampling-structured-decoding", "title": "采样器与结构化解码", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["采样", "结构化输出", "推理", "投机解码"], "desc": "从稳定 softmax、temperature 与 top-p 推导采样，再把 token 级语法状态、EOS 和回滚接入生成循环，配数值交互、四道练习及 JSON 实验。"},
    {"slug": "inference-benchmarking", "title": "推理压测与 SLO：测出真正可用的容量", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["推理", "性能分析", "SLO", "排队"], "desc": "用请求时间线推导 TTFT、TPOT、ITL 和 goodput，理解开放负载、客户端排队与可复现的推理评测。"},
    {"slug": "prefill-decode-disaggregation", "title": "Prefill / Decode 分离：把计算与缓存交接清楚", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["推理", "并行", "通信", "KV cache"], "desc": "从 KV 传输字节、P/D 时间线和固定 GPU 容量分析分离推理，推导队列、交接协议与多轮会话取舍。"},
    {"slug": "lora-qlora", "title": "LoRA / QLoRA：参数高效微调", "date": "2026-10-20", "kind": "interactive", "cat": "recipes", "tags": ["LoRA", "QLoRA", "微调", "量化", "显存"], "desc": "从低秩前向与梯度推导到 NF4 账本、checkpoint 梯度排查、保存合并和量化误差；含数值图解、四道习题与 NumPy 实验。"},
    {"slug": "multi-adapter-serving", "title": "多 Adapter 推理服务", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["LoRA", "推理", "调度", "KV cache", "显存"], "desc": "从分组矩阵乘法、异构 rank 与显存准入，到 GPU 权重寿命、版本切换、KV 隔离和冷请求排错；含图解、四道习题与 CPU 实验。"},
    {"slug": "multimodal-inference", "title": "图像与视频推理：token、计算与显存账本", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["多模态", "推理", "KV cache", "视频", "显存"], "desc": "沿像素、patch、视觉token与语言KV推导图像视频成本，分析动态batch、两级缓存、特征占位不变量与资源调度；含图解、四道习题与实验。"},
    {"slug": "vla-inference", "title": "VLA：从视觉 token 到机器人动作", "date": "2026-10-20", "kind": "interactive", "cat": "models", "tags": ["VLA", "多模态", "机器人", "推理", "延迟"], "desc": "从动作归一化、离散token与flow matching，推导条件KV复用、计算供给、动作队列和观察年龄；结合坐标变换、闭环误差、习题与仿真。"},
    {"slug": "deepseek-v4-flash", "title": "DeepSeek-V4-Flash 架构：一个 token 的旅程", "date": "2026-10-20", "kind": "interactive", "cat": "models", "tags": ["DeepSeek", "架构", "MoE", "稀疏注意力"], "desc": "从真实张量形状推导 mHC、共享 KV、压缩因果边界与专家路由，结合可操作图解、移植排错、计算题和 NumPy 实验。"},
    {"slug": "deepseek-v4-flash-inference", "title": "DeepSeek-V4-Flash 推理：一枚 token，4 张卡", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["DeepSeek", "推理", "KV cache", "并行", "通信"], "desc": "逐卡推导原生 FP4 权重、复制缓存、prefill 峰值、FP32 归约与专家调度；用可操作容量账本和时间线连接工程排错、习题与 CPU 实验。"},
    {"slug": "deepseek-v4-pro", "title": "DeepSeek-V4-Pro 架构：一个 token 的旅程", "date": "2026-10-20", "kind": "interactive", "cat": "models", "tags": ["DeepSeek", "架构", "MoE", "稀疏注意力"], "desc": "从真实张量形状推导 mHC、共享 KV、压缩因果边界与专家路由，结合可操作图解、移植排错、计算题和 NumPy 实验。"},
    {"slug": "deepseek-v4-pro-inference", "title": "DeepSeek-V4-Pro 推理：一枚 token，8 张卡", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["DeepSeek", "推理", "KV cache", "并行", "通信"], "desc": "逐卡推导原生 FP4 权重、复制缓存、prefill 峰值、FP32 归约与专家调度；用可操作容量账本和时间线连接工程排错、习题与 CPU 实验。"},
    {"slug": "kv-cache", "title": "KV cache：记住已经算过的历史", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["KV cache", "推理", "显存"], "desc": "从一行 attention 的数值计算出发，推导 KV 为什么可复用、prefill/decode 怎样交接、位置与 mask 如何对齐，再算清 GQA 缓存的容量和读取成本；含可操作图解、符号释义、习题与 NumPy 实验。"},
    {"slug": "kv-cache-compression", "title": "KV cache 压缩：把历史装得更小", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["KV cache", "低精度", "长上下文"], "desc": "按状态形状、位置数和位宽推导 GQA、MLA 与 KV 量化，计算分组元数据、residual 和读取成本，附工程排查、习题与 NumPy 实验。"},
    {"slug": "sparse-attention", "title": "Sparse attention：只看需要的历史", "date": "2026-10-20", "kind": "interactive", "cat": "models", "tags": ["注意力", "稀疏注意力", "长上下文"], "desc": "推导稀疏归一化、窗口配对数、内容选择与块布局，结合 mask 构建、块利用率与端到端性能排查，附交互图解、练习与实验。"},
    {"slug": "linear-attention", "title": "Linear attention：把历史写进状态", "date": "2026-10-20", "kind": "interactive", "cat": "models", "tags": ["线性注意力", "架构", "长上下文"], "desc": "从核 attention 推导递推、归一化、门控、Delta rule 与分块计算；结合 padding、状态复用和精度问题，附数值图解、习题与实验。"},
    {"slug": "paged-attention", "title": "PagedAttention：让 KV cache 按需生长", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["KV cache", "显存", "推理"], "desc": "从连续缓存的浪费讲到页表寻址、分块 softmax、容量预算和写时复制：用完整推导、三个可操作动态图、四道习题与 CPU 实验理解分页推理。"},
    {"slug": "continuous-batching", "title": "Continuous batching：每一轮重新组队", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["调度", "推理", "KV cache"], "desc": "从一轮调度的 token 预算推导 chunked prefill、KV 准入和公平性，手算 TTFT、ITL 与吞吐，用时间轴和预算实验定位长尾与抢占。"},
    {"slug": "prefix-caching", "title": "Prefix caching：相同的开头，只算一次", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["KV cache", "推理", "调度"], "desc": "推导前缀 KV 的计算身份、哈希链和满块边界，量化重算量与 TTFT 收益，分析 tokenizer 差异、共享池容量、缓存驱逐和命中后的性能问题。"},
    {"slug": "speculative-decoding", "title": "投机解码", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["投机解码", "推理"], "desc": "从接受概率与残差分布证明投机采样，推导有效输出长度和加速条件，追踪验证行、KV 回滚与纠正 token，并定位高接受率却变慢的工程问题。"},
    {"slug": "dspark", "title": "DSpark", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["投机解码", "DeepSeek", "调度"], "desc": "拆开 DSpark 的并行 backbone、低秩 Markov head、条件置信度和硬件感知调度，手算校准、预算与因果依赖，理解草稿质量怎样转化成服务吞吐。"},
    {"slug": "llm-quantization", "title": "LLM 量化", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["量化", "低精度", "显存"], "desc": "从整数码、打包与元数据到 GPTQ、AWQ、SmoothQuant，连接输出误差与实际 kernel 的带宽、计算和对齐，配数值图解、四道习题与实验。"},
    {"slug": "distributed-inference", "title": "分布式推理：数据怎样走过每张卡", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["推理", "并行", "通信", "MoE"], "desc": "从分片矩阵乘法、TP 通信到 PP、EP 与显存，推导多卡推理的部署取舍，配交互算例和实验。"},
    {"slug": "cpu-overhead", "title": "CPU overhead：GPU 为什么在等", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["推理", "调度", "profiling"], "desc": "用异步队列递推、同步语义与 trace 解释 GPU 空闲，区分提交、排队和执行时间，再计算优化上限与尾延迟；含交互时间线、习题和 CPU 排程实验。"},
    {"slug": "cuda-graph", "title": "CUDA Graph：把一轮计算录下来", "date": "2026-10-20", "kind": "interactive", "cat": "kernels", "tags": ["GPU", "kernel", "推理"], "desc": "从捕获与固定存储推导 replay、输出别名、KV 元数据更新和形状桶，计算空算、持久显存与回本点；含交互图、工程排查、习题及 CPU/CUDA 实验。"},
    {"slug": "flash-attention", "title": "FlashAttention 1、2、3", "date": "2026-10-20", "kind": "interactive", "cat": "kernels", "tags": ["GPU", "kernel", "注意力"], "desc": "从 HBM 流量与在线 softmax 推导分块前反向，再理解 FA2 工作划分与 FA3 异步流水，结合 mask、硬件和性能排查，附习题与实验。"},
    {"slug": "rmsnorm-kernel", "title": "RMSNorm 与 GPU 算子优化", "date": "2026-10-19", "kind": "interactive", "cat": "kernels", "tags": ["GPU", "kernel", "RMSNorm"], "desc": "从 RMSNorm 前反向推导到 FP32 归约、寄存器、向量化与融合，分析精度契约、资源瓶颈和带宽账本；含可操作图、计算题、Triton 核心与 NumPy 梯度实验。"},
    {"slug": "gradient-checkpointing", "title": "Gradient checkpointing", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["显存", "激活重算"], "desc": "从反向依赖推导激活寿命、最优分段和选择性重算，排查冻结输入、RNG、FSDP 与反向峰值；含动态图、计算题和 NumPy/PyTorch 实验。"},
    {"slug": "optimizers", "title": "优化器：从 SGD 到 Muon", "date": "2026-10-19", "kind": "interactive", "cat": "recipes", "tags": ["优化器"], "desc": "从曲率和动量推导AdamW、Lion、Adafactor、Shampoo/SOAP与Muon，连接状态显存、更新尺度和GPU执行；保留交互地形，加入计算题与数值实验。"},
    {"slug": "deepseek-v4", "title": "DeepSeek-V4 技术报告精读", "date": "2026-10-19", "kind": "interactive", "cat": "models", "tags": ["DeepSeek", "架构", "稀疏注意力", "MoE", "长上下文", "优化器", "低精度"], "desc": "拆解 DeepSeek-V4：CSA/HCA 注意力、mHC、Muon、FP4 QAT 与多教师蒸馏，附成本计算器。"},
    {"slug": "qwen3-next", "title": "Qwen3-Next：混合线性注意力与超稀疏 MoE", "date": "2026-10-19", "kind": "interactive", "cat": "models", "tags": ["Qwen", "架构", "线性注意力", "MoE", "投机解码"], "desc": "推导 Gated DeltaNet、混合注意力与 512 expert MoE，追踪前缀快照和 MTP 回滚，计算缓存、专家批量与跨卡通信；附数值交互、六道习题及 NumPy 实验。"},
    {"slug": "kimi-k3", "title": "Kimi K3：KDA、注意力残差与 Stable LatentMoE", "date": "2026-10-19", "kind": "interactive", "cat": "models", "tags": ["Kimi", "架构", "线性注意力", "MoE", "优化器", "长上下文"], "desc": "精读 Kimi K3 技术报告：有下界的 KDA、注意力残差、Stable LatentMoE 与分位数均衡，以及 1M 上下文的训练、RL 与推理系统。"},
    {"slug": "kimi-k2", "title": "Kimi K2：MuonClip 与万亿参数 Agent 模型", "date": "2026-10-19", "kind": "interactive", "cat": "models", "tags": ["Kimi", "架构", "优化器", "MoE", "RL"], "desc": "逐节精读 Kimi K2 技术报告：Muon 与 QK-Clip、数据改写、超稀疏 MoE、训练基础设施、Agent 数据合成与 RL。"},
    {"slug": "kimi-linear", "title": "Kimi Linear：Kimi Delta Attention", "date": "2026-10-19", "kind": "interactive", "cat": "models", "tags": ["Kimi", "线性注意力", "架构", "KV cache"], "desc": "从线性注意力、DeltaNet、Gated DeltaNet 推到 KDA：通道级门控、分块并行算法、3:1 混合与 NoPE，以及效率实验。"},
    {"slug": "smol-decisions", "title": "从目标到决策：训练什么、怎么做消融", "date": "2026-10-19", "kind": "interactive", "cat": "recipes", "tags": ["消融", "评测"], "desc": "训练大模型之前的决策方法：该不该训、训什么、怎样用小规模消融验证每个决定，以及评测噪声、消融成本与 scaling law。"},
    {"slug": "smol-architecture", "title": "架构选择：注意力、位置编码、MoE 与混合模型", "date": "2026-10-19", "kind": "interactive", "cat": "recipes", "tags": ["架构", "注意力", "MoE", "KV cache"], "desc": "现代 LLM 的结构选择：MHA/GQA/MLA 与 KV cache、文档掩码、RoPE 与 NoPE、稳定性技巧、MoE 设计与混合模型，以及怎么在三者之间选。"},
    {"slug": "smol-optim-data", "title": "分词器、优化器、超参与数据配比", "date": "2026-10-19", "kind": "interactive", "cat": "recipes", "tags": ["优化器", "数据"], "desc": "预训练的另一半决定：BPE 与分词器评估、AdamW 与 Muon、学习率调度、batch size 与超参 scaling law、数据配比与多阶段训练。"},
    {"slug": "smol-marathon", "title": "训练马拉松：吞吐、loss spike 与中途调整", "date": "2026-10-19", "kind": "interactive", "cat": "recipes", "tags": ["稳定性", "长上下文"], "desc": "按下开始键之后会发生什么：吞吐与完成时间的账、起跑检查清单、SmolLM3 训练中的四个谜案、loss spike 的诊断与处理、多阶段 mid-training 与 4k→128k 长上下文扩展。"},
    {"slug": "smol-post-training", "title": "后训练：SFT、偏好优化与 RL", "date": "2026-10-19", "kind": "interactive", "cat": "recipes", "tags": ["后训练", "RL"], "desc": "把 3B 基座变成混合推理助手：评测先行、chat template 与 loss 掩码、SFT 超参与 packing、推理 mid-training、DPO/APO 推导、GRPO 与长度惩罚、on-policy 蒸馏。"},
    {"slug": "smol-infrastructure", "title": "基础设施：GPU 集群、通信与存储", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["硬件", "通信", "GPU"], "desc": "训练大模型的基础设施：H100 内部、PCIe/NVLink/EFA 实测带宽、nccl-tests、存储分层、节点健康与 checkpoint，以及需要多少张 GPU。"},
    {"slug": "scaling-memory", "title": "训练的账本：显存、FLOPs 与 MFU", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["显存"], "desc": "一步训练里显存花在哪、FLOPs 怎么数、MFU 怎么算：从 16 字节/参数和激活公式推到 6ND，附显存计算器与训练时长估算器。"},
    {"slug": "scaling-roofline", "title": "Roofline：算力、带宽与算术强度", "date": "2026-10-19", "kind": "interactive", "cat": "kernels", "tags": ["硬件", "GPU"], "desc": "一个运算到底卡在算力、显存带宽还是网络上：算术强度、roofline 图、矩阵乘法的临界 batch，以及 TPU 与 GPU 的硬件结构。"},
    {"slug": "scaling-collectives", "title": "集合通信与分片矩阵乘法", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["通信", "并行"], "desc": "从 ring 字节账本和延迟模型，到 device mesh、分片矩阵乘法与梯度通信，保留可操作的通信与布局探索器。"},
    {"slug": "scaling-data-parallel", "title": "数据并行、ZeRO 与 FSDP", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["并行", "显存", "通信"], "desc": "从朴素数据并行讲到 ZeRO-1/2/3 与 FSDP：通信重叠、分桶、no_sync、显存公式，以及什么时候会被通信卡住。"},
    {"slug": "scaling-tensor-parallel", "title": "张量并行与序列并行", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["并行", "通信"], "desc": "从矩阵乘法的两种切法讲起：Megatron 张量并行、f/g 算子、词表并行、序列并行、通信量与为什么 TP 不出节点。"},
    {"slug": "scaling-context-parallel", "title": "上下文并行与 Ring Attention", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["并行", "长上下文", "注意力"], "desc": "长序列训练的激活怎么切：online softmax 合并、Ring Attention、zig-zag 负载均衡、All-Gather 式 CP 与 DeepSpeed-Ulysses。"},
    {"slug": "scaling-pipeline", "title": "流水线并行：从 GPipe 到 DualPipe", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["并行"], "desc": "从 micro-batch 依赖推导流水线空泡、激活峰值和通信成本，交互比较 AFAB、1F1B、交错、Zero Bubble 与 DualPipe。"},
    {"slug": "scaling-moe", "title": "MoE 与专家并行", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["MoE", "并行", "通信"], "desc": "从路由器讲起：top-k、softmax 与 sigmoid 门控、容量与丢 token、负载均衡损失的推导、z-loss、细粒度与共享专家、无辅助损失均衡，再到专家并行的 all-to-all、通信量和与 DP/TP/PP 的组合。"},
    {"slug": "scaling-5d", "title": "5D 并行：组合与选型", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["并行"], "desc": "DP、TP、CP、PP、EP 各切什么、通信放在网络哪一层、怎么在设备网格上组合，以及一套选配置的流程和一个配置规划器。"},
    {"slug": "scaling-jax", "title": "JAX 分片编程与 Profiling", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["JAX", "并行", "profiling"], "desc": "用 JAX 的 mesh、PartitionSpec、shard_map 写分片程序，读懂 HLO 和 profile，在时间线上分出计算、通信与空闲。"},
    {"slug": "scaling-gpu-kernels", "title": "GPU 内核与 FlashAttention", "date": "2026-10-19", "kind": "interactive", "cat": "kernels", "tags": ["GPU", "kernel", "注意力"], "desc": "从 SM、warp、内存层级讲起，到访存合并、分块、融合、Triton，再到 FlashAttention 1–3 的推导与实现。"},
    {"slug": "scaling-mixed-precision", "title": "混合精度训练：从 BF16 到 FP8 与 NVFP4", "date": "2026-10-19", "kind": "interactive", "cat": "kernels", "tags": ["低精度", "量化"], "desc": "从浮点编码到三次训练 GEMM，推导 FP8/FP4 的缩放、转置一致性、随机舍入和 Adam 噪声，分析分布式量化通信与显存账，附数值交互、习题及 NumPy 实验。"},
    {"slug": "scaling-inference", "title": "Transformer 推理的数学", "date": "2026-10-19", "kind": "interactive", "cat": "inference", "tags": ["推理", "KV cache", "投机解码", "并行"], "desc": "从 KV cache 到投机解码：推导带宽与算力瓶颈、延迟下界、吞吐权衡和多卡服务账本，结合数值计算与服务排查。"},
    {"slug": "next-token", "title": "一次只猜一个 token", "date": "2026-10-19", "kind": "interactive", "cat": "basics", "tags": ["推理"], "desc": "从 tokenizer、logits 与采样推导生成循环，跟踪输入和 KV cache 的交接，结合 padding 排错、交互图解与练习。"},
    {"slug": "hello-world", "title": "新博客，从零开始", "date": "2026-10-19", "kind": "interactive", "cat": "basics", "tags": [], "desc": "这个博客怎么搭的，以及一个可以拖动的 temperature 采样演示。"},
  ];

  window.SITE = { POSTS, CATEGORIES, KINDS };

  // ---------- post pages: tags under the header, related posts at the end ----------
  const slug = location.pathname.match(/^\/posts\/([^/]+)\//)?.[1];
  const me = POSTS.find((p) => p.slug === slug);
  if (!me) return;
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const tagLink = (t) => '<a class="tag" href="/blog/?tag=' + encodeURIComponent(t) + '">#' + esc(t) + '</a>';

  function render() {
    const meta = document.querySelector('.post-header .post-meta');
    if (meta && !meta.querySelector('.post-tax')) {
      const span = document.createElement('span');
      span.className = 'post-tax';
      span.innerHTML = '<a class="cat" href="/blog/?cat=' + me.cat + '">' + esc(CATEGORIES[me.cat]) + '</a>' + me.tags.map(tagLink).join('');
      meta.appendChild(span);
    }
    const main = document.querySelector('main.wrap');
    if (!main) return;
    const links = window.POST_LINKS || {};
    const by = (s) => POSTS.find((p) => p.slug === s);
    const out = (links[slug] || []).map(by).filter(Boolean);
    const inn = Object.keys(links).filter((s) => s !== slug && links[s].includes(slug)).map(by).filter(Boolean);
    const seen = new Set([slug, ...out.map((p) => p.slug), ...inn.map((p) => p.slug)]);
    const near = POSTS.filter((p) => !seen.has(p.slug))
      .map((p) => ({ p, n: p.tags.filter((t) => me.tags.includes(t)).length + (p.cat === me.cat ? 0.5 : 0) }))
      .filter((x) => x.n >= 1).sort((a, b) => b.n - a.n).slice(0, 4).map((x) => x.p);
    const item = (p) => '<li><a href="/posts/' + p.slug + '/">' + esc(p.title) + '</a> <span class="' + KINDS[p.kind][1] + '">' + KINDS[p.kind][0] + '</span></li>';
    const group = (title, list) => (list.length ? '<div><h3>' + title + '</h3><ul>' + list.map(item).join('') + '</ul></div>' : '');
    const html = group('本文引用', out) + group('引用本文', inn) + group('相关主题', near);
    if (!html) return;
    const box = document.createElement('section');
    box.className = 'related';
    box.innerHTML = '<h2>相关文章</h2><div class="related-grid">' + html + '</div>';
    main.appendChild(box);
  }
  // Hover / focus a link to another post: show its title, kind and summary.
  function previews() {
    const card = document.createElement('div');
    card.className = 'link-card';
    card.hidden = true;
    document.body.appendChild(card);
    let hideTimer = 0;
    document.querySelectorAll('.prose a[href^="/posts/"]').forEach((a) => {
      const s = a.getAttribute('href').split('/')[2];
      const p = POSTS.find((x) => x.slug === s);
      if (!p || s === slug) return;
      const show = () => {
        clearTimeout(hideTimer);
        card.innerHTML = '<span class="' + KINDS[p.kind][1] + '">' + KINDS[p.kind][0] + '</span> <span class="lc-cat">' + esc(CATEGORIES[p.cat]) + '</span>' +
          '<b>' + esc(p.title) + '</b><span class="lc-desc">' + esc(p.desc) + '</span>';
        card.hidden = false;
        const r = a.getBoundingClientRect();
        const w = Math.min(340, innerWidth - 24);
        card.style.width = w + 'px';
        card.style.left = Math.max(12, Math.min(innerWidth - w - 12, r.left)) + 'px';
        const below = r.bottom + 8;
        card.style.top = (below + card.offsetHeight > innerHeight ? r.top - card.offsetHeight - 8 : below) + 'px';
      };
      const hide = () => { hideTimer = setTimeout(() => { card.hidden = true; }, 120); };
      a.addEventListener('mouseenter', show);
      a.addEventListener('focus', show);
      a.addEventListener('mouseleave', hide);
      a.addEventListener('blur', hide);
    });
    addEventListener('scroll', () => { card.hidden = true; }, { passive: true });
  }

  const start = () => { render(); if (matchMedia('(hover: hover)').matches) previews(); };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
