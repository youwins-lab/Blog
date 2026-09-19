---
layout: post
title: "AI 칩 아키텍처 여섯 갈래 — NVIDIA·TPU·AMD·Cerebras·Trainium·Groq가 memory wall을 넘는 법"
date: 2026-09-19
categories: study hardware ai-chip-architectures
tags: [NVIDIA, TPU, AMD, Cerebras, Trainium, Groq, GPU, Networking, LLMSO]
---

# AI 칩 아키텍처 여섯 갈래 — NVIDIA·TPU·AMD·Cerebras·Trainium·Groq가 memory wall을 넘는 법

> NVIDIA GPU, Google TPU, AMD Instinct, Cerebras WSE, AWS Trainium, Groq LPU. 여섯 갈래의 아키텍처가 각자 memory wall을 어떻게 공격하는지 정리했다. 축은 철학 / 연산 / 메모리 / numerics / scale-up·scale-out / software stack 여섯 개로 고정하고 여섯 칩을 같은 틀에 놓고 읽었다. 원문은 Jacob Peake의 ["AI Chip Architectures"](https://www.jepeake.com/ai-chip-architectures)이고, 이 글은 LLMSO 강의 마지막 과제로 그 원문을 정리·해석한 것이다.

## 0. 왜 이 글인가

2018년 ISCA에서 Hennessy와 Patterson이 튜링 강연 "A New Golden Age for Computer Architecture"를 했다. 1980년대 단일 스레드 CPU 성능은 연 52%씩 올랐는데, Moore's Law와 Dennard scaling이 끝난 2018년에는 연 3%였다. 그래서 domain-specific architecture(DSA)가 필요하다는 게 논지였고, 사례로 든 게 이미 양산 중이던 TPU v1이었다 — CPU 대비 추론 throughput 29배, 전력효율 80배. 강연의 마지막 예측은 "다음 10년은 새로운 컴퓨터 아키텍처의 캄브리아기 대폭발이 될 것"이었다.

그 예측은 맞았다. 그리고 그중 실제 대규모 배치에 성공한 건 사실상 네 부류다.

- **GPU** — NVIDIA, AMD
- **systolic array 가속기** — Google TPU, AWS Trainium
- **wafer-scale engine** — Cerebras
- **LPU** — Groq

2026년 현재 판세는 원문 기준으로 이렇다. NVIDIA가 확실한 선두, AMD가 추격 중이다(OpenAI·Meta가 각각 6 GW 규모로 약정). TPU는 Gemini를 학습시키고 Anthropic에 최대 100만 칩 규모로 공급 예정인데, 동시에 Anthropic은 Trainium 100만 칩 이상에서도 Claude를 돌린다. Cerebras는 OpenAI 추론을 서빙하고, Groq LPU는 200억 달러 규모 acquihire로 NVIDIA에 흡수됐다.

칩 하나를 이해한다는 건 결국 네 가지 질문에 답하는 것이다. ① 데이터가 어디에 사는가 ② 연산 유닛까지 어떻게 움직이는가 ③ 연산 유닛의 모양은 무엇인가 ④ 칩끼리 어떻게 대화하는가. 아래 여섯 섹션 전부 이 순서로 읽으면 된다.

## 1. 문제 정의 — 진짜 적은 FLOPS가 아니라 memory wall

### 1-1. AI 연산 = matrix multiplication

Transformer는 matmul의 연속이다. Q/K/V projection → attention → output projection → FFN, 그 사이사이에 normalization·activation·residual add 같은 element-wise 연산이 끼어든다. frontier 모델 학습 한 번에 대략 10²⁵ multiply-accumulate가 들어간다.

### 1-2. 그런데 matmul의 모양이 워크로드마다 다르다

| 단계 | 연산 형태 | arithmetic intensity | 병목 |
| --- | --- | --- | --- |
| Training | GEMM (matrix × matrix) | 높음 | compute-bound |
| Prefill | GEMM — 입력 시퀀스 전체를 한 번에 통과 | 높음 | compute-bound |
| Decode | GEMV (matrix × vector) | 수 자릿수 하락 | memory-bandwidth-bound |

decode가 아픈 이유는 구조적이다. autoregressive라서 토큰 N이 나오기 전엔 N+1을 시작할 수 없고, 한 스텝에 딱 한 토큰만 projection되므로 모든 matmul이 GEMV로 떨어진다. 토큰 하나를 뽑기 위해 모델 전체 weight를 한 번 다 읽고, attention을 위해 KV cache도 전부 읽어야 한다.

### 1-3. 그래서 서빙 시스템은 GEMV를 GEMM으로 되돌리려 애쓴다

- **continuous batching** — 여러 사용자의 decode 스텝을 쌓아 올린다
- **speculative decoding** — draft 토큰 K개를 쌓아 한 번에 검증
- **multi-token prediction** — 같은 트릭을 모델 내부에 접어 넣음

덕분에 matmul unit 활용률과 Ops/Byte가 올라간다. 다만 continuous batching에서도 KV cache는 요청별로 각자 읽어야 하므로, long-context decode는 weight-bandwidth-bound에서 KV-bandwidth-bound로 병목이 이동한다.

연산량은 지수적으로 늘었는데 메모리 대역폭은 그러지 못했다. 결국 아키텍처 문제는 "숫자를 matmul이 일어나는 곳까지 충분히 빨리 옮기는 법"으로 환원된다. 이게 memory wall이고, 아래 여섯 회사는 이 데이터 이동 게임을 각기 다른 방식으로 이기려 한다.

---

## 2. NVIDIA GPU — massively parallel processor

수천 개 스레드를 가진 프로그래머블 칩을 host CPU가 지휘하고 CUDA로 노출하는 것, 이게 NVIDIA가 병렬 워크로드에 내놓은 답이다. 세대마다 가속 primitive를 SM 위에 얹되 프로그래밍 모델은 건드리지 않는다. 같은 칩이 학습·추론·그래픽·과학계산을 다 한다(accelerated computing).

### 계보

| 연도 | 아키텍처 | 핵심 |
| --- | --- | --- |
| 2006 | Tesla (G80) | 최초 CUDA GPU, SIMT 실행 모델 |
| 2016 | Pascal (P100) | NVLink 1.0, HBM2, native FP16 — 첫 딥러닝 전용 설계 |
| 2017 | Volta (V100) | 최초 Tensor Core, independent thread scheduling |
| 2020 | Ampere (A100) | TF32, structured sparsity(2:4), MIG |
| 2022 | Hopper (H100) | FP8, Transformer Engine, HBM3, TMA, thread block cluster, async `wgmma` |
| 2024 | Blackwell (B200) | FP4, TMEM, 2-die chiplet GPU, NVLink 5 |
| 2025 | Blackwell Ultra (B300) | FP4 ~1.5배, HBM3e 288 GB — long-context reasoning 대응 |
| 2026 | Rubin | HBM4, 3rd-gen Transformer Engine, Vera CPU, Rubin CPX로 prefill 분리 |
| 2027 | Rubin Ultra | 4-die 패키지, 패키지당 HBM4e 1 TB, 600 kW NVL576 Kyber rack |

### 연산 — matmul 명령의 20년 진화사

SM 개수는 V100 80 → A100 108 → H100 132 → B200 148 → B300 160 → Rubin 224로 늘었다. 그런데 SM 내부 레시피는 안 변한다. 4개의 sub-partition이 각자 warp scheduler·dispatch·레지스터 파일·CUDA Core lane·SFU를 갖고, Tensor Core로 가는 전용 포트를 하나씩 쥔다. 32스레드 warp가 SIMT lock-step으로 돌고, resident warp를 바꿔 끼우며 stall을 숨긴다.

Transformer 블록의 FLOPs 중 약 99%가 matmul이라 실질 throughput은 전부 Tensor Core에서 나온다. Tensor Core는 타일 단위 fused MMA(`D = A·B + C`)를 수행하고, K축을 돌면서 accumulator에 partial sum을 접어 넣는다.

여기서 중요한 건 "누가 이 명령을 issue하느냐"의 변화다.

| 세대 | 명령 | issue 주체 | operand 위치 | 동작 |
| --- | --- | --- | --- | --- |
| Volta | `mma.sync` | warp 32스레드 | A/B/D 전부 register | 동기 — 완료까지 warp block |
| Hopper | `wgmma.mma_async` | warp-group 128스레드 | B는 SMEM descriptor | 즉시 반환, 백그라운드 실행 |
| Blackwell | `tcgen05.mma` | 단일 스레드 | A·B는 descriptor, D는 TMEM | 즉시 반환, `mbarrier`로 완료 신호 |

즉 matmul은 더 커지면서 동시에 issue하는 스레드에게는 더 가벼워졌다. 32 lane이 lock-step으로 움직이던 명령이 이제는 사실상 descriptor 하나짜리 커맨드가 됐다. Blackwell에는 두 SM이 짝을 이뤄 256×256×16 타일을 처리하는 CTA-pair 변형도 있다.

이 decoupling이 attention kernel 효율의 핵심이다. matmul이 백그라운드로 도는 동안 warp는 softmax를 돌리거나 mask를 씌우거나 다음 타일을 미리 로드한다. FlashAttention-3, FA4의 구조 자체가 이 overlap 위에 서 있다.

### 메모리 — hardware-managed cache + software hint

- **HBM**: V100 32 GB HBM2 → H100 80 GB → B200 192 GB HBM3e → B300 288 GB → Rubin 288 GB HBM4
- **L2**: V100 6 MB → A100 40 MB → H100 50 MB → B200 60 MB (2-die에 30 MB씩, hot tile을 가까운 die에 pin 가능)
- **L1/SMEM**: SM당 256 KB, kernel launch 시점에 L1 / scratchpad 비율 결정
- **TMEM** (Blackwell 신설): SM당 256 KB, MMA accumulator 전용, Tensor Core만 주소 지정 가능

데이터 이동도 warp에서 계속 떼어냈다. Ampere의 `cp.async`(register 우회 HBM→SMEM) → Hopper의 TMA(전용 DMA 엔진, 스레드 하나가 multi-dim tile descriptor만 제출하면 주소 계산을 엔진이 처리, cluster-level multicast로 HBM 1회 읽기를 여러 SM에 fan-out) → Blackwell은 TMEM으로 직접 로드. 세대마다 "warp가 타일당 해야 할 일이 하나씩 줄어드는" 궤적이다.

한 block 안에서 일부 warp는 producer(TMA 로드 연발), 일부는 consumer(`wgmma` 발사) 역할을 한다. 동기화는 block 전체 `__syncthreads()`가 아니라 `mbarrier` 기반 fine-grained handshake다. FlashAttention-3, CUTLASS ping-pong GEMM, Blackwell FA4가 전부 이 레시피를 쓴다.

### Numerics

FP32 → FP16(+FP32 accumulate, loss scaling) → TF32/BF16/2:4 sparsity → FP8(E4M3·E5M2 + Transformer Engine 자동 스케일링) → FP4 + microscaling MX format(block 단위 shared exponent) → Rubin의 NVFP4·native FP6. 세대마다 비트를 반으로 자르고, 더 촘촘한 scaling으로 정확도를 되사오는 방식으로 와트당 약 2배를 번다. 여기에 패키징이 합류한다 — B100/B200/B300은 reticle 한계 die 2개를 약 10 TB/s `NV-HBI`로 꿰매 소프트웨어에는 GPU 하나로 보인다.

### Scaling

scale-up은 여러 GPU를 하나의 coherent memory domain으로 묶는 것이다. 어떤 GPU든 NVLink로 다른 GPU의 HBM을 직접 load/store하고, 단일 주소 공간에 명시적 전송이 없어 레이턴시가 나노초 단위다. scale-out은 그 domain들을 rack·cluster 레벨로 네트워킹하는 것이다. 주소 공간이 분리되고 명시적 RDMA로 건너가며, 레이턴시는 마이크로초 단위인 대신 클러스터당 수만 칩까지 늘어난다.

대역폭을 많이 먹는 collective(tensor parallelism, MoE expert routing)는 scale-up 안에 가둬두고, data parallelism과 pipeline parallelism이 scale-out fabric을 건넌다. 이 분업 규칙은 여섯 아키텍처 전부에 공통이다.

NVLink는 cache-coherent fabric이지만 본질적으로 point-to-point다. NVSwitch가 crossbar 역할을 해서 모든 GPU가 동시에 full NVLink 대역으로 all-to-all 통신한다(non-blocking).

- **HGX** 8-GPU baseboard — H100 SXM 8장 + x86 host(PCIe Gen5)
- **GH200** — Grace 1 + H100 1을 NVLink-C2C 900 GB/s로 접합, PCIe host-device hop 제거
- **GB200 NVL72** — GB200 모듈 36개 = GPU 72 + Grace 36, HBM 13.5 TB + LPDDR5X 17 TB를 하나의 평평한 coherent 주소 공간으로
- **NVL144** (2026) — 같은 Oberon rack에 Rubin 패키지 72개(새 die 카운팅으로 144 GPU), HBM4 + NVLink 6
- **NVL576** (2027) — 4-die Rubin Ultra 패키지 144개를 새 Kyber 섀시에, 576 GPU die가 하나의 domain

passive copper가 rack-as-one-GPU를 가능하게 한다. NVL72의 NVLink fabric은 backplane에 blind-mate된 케이블 5,184가닥(rack당 약 2마일, in-cable retimer 없음, SerDes는 GPU·switch ASIC에 내장) 위를 달리며 72 GPU에 걸쳐 약 130 TB/s all-to-all을 나른다. NVIDIA 추산으로 광 대비 rack당 약 20 kW를 절약한다. 2m 미만 구간에서는 copper가 전력·비용·signal integrity에서 여전히 이기고, 그 너머부터는 유리로 가야 한다. NVL576도 이 선을 지키려고 랙 자체를 다시 설계했다 — Kyber는 Oberon의 약 두 배 높이로, 576 die 전부를 passive copper 도달 거리 안에 욱여넣도록 치수를 잡았다.

scale-out fabric은 coherent하지 않다. 노드마다 주소 공간이 따로고, 소프트웨어가 시작하는 명시적 RDMA로만 건너간다(보통 NCCL의 all-reduce / all-to-all로 감싸서). 레퍼런스 클러스터는 DGX SuperPOD다 — NVL72 8랙을 Quantum-X800 InfiniBand로 묶어 Blackwell 576 GPU를 하나의 스케줄러 밑에 둔다. Rubin SuperPOD은 같은 8랙 패턴에 NVL144라 1,152 GPU다.

GPU마다 ConnectX NIC가 붙는다. Blackwell은 ConnectX-8 800 Gbps/GPU인데, 이건 per-GPU NVLink보다 한 자릿수 낮고 레이턴시는 ns에서 μs로 뛴다. Rubin은 ConnectX-9 1.6 Tbps/GPU. NIC 옆엔 BlueField DPU가 붙어 storage·networking·security를 offload한다. InfiniBand 대신 Ethernet을 원하는 고객에겐 Spectrum-X(AI 트래픽에 맞춘 lossless Ethernet)가 있다.

copper→glass 전환점은 rack 경계다. 200G/lane에서 passive copper DAC는 대략 1.5~2 m가 한계라 rack을 건너는 순간 광이다. 오늘의 SuperPOD spine은 OSFP-RHS pluggable transceiver 위를 달리는데, 모듈마다 laser·modulator·photodetector·DSP가 들어 있다. 수천 GPU로 뻗는 spine이면 pluggable 수만 개, laser 전력만 수십 kW다.

Rubin 세대에서 이 광 레이어가 switch ASIC 안으로 접혀 들어간다. Quantum-X Photonics(IB)와 Spectrum-X Photonics(Ethernet)가 TSMC COUPE로 co-packaged optics를 구현하는데, NVIDIA 주장으로 laser 수 약 1/4, 링크 전력 약 1/3.5다. GPU를 2-die로 만들고 HBM을 옆에 쌓았던 chiplet 논리가 이제 네트워크 계층에 나타난 셈이다. 별도로 NVLink Fusion은 third-party CPU·XPU가 NVLink domain에 합류할 길을 열었다.

### Software — 해자는 CUDA가 아니다

CUDA 추상화는 18년째 거의 그대로다. 2007년 이후 작성된 어떤 CUDA kernel도 Blackwell에서 컴파일되고 돈다. 이 연속성은 해자이자 제약이다 — 너무 많은 코드가 의존하기 때문에 NVIDIA는 SM을 근본적으로 다시 생각할 수 없다. 대신 CUDA 소프트웨어 투자는 세대를 건너 복리로 쌓인다.

PTX/SASS 위로 20년간 쌓인 스택이 있다. cuBLAS·cuDNN, CUTLASS(템플릿 C++에 GEMM 노하우를 인코딩), TensorRT-LLM(paged attention, in-flight batching, speculative decoding), PyTorch·Triton·JAX 바인딩. FlashAttention은 FA1~FA4 각 세대가 최신 NVIDIA 실리콘에 손으로 튜닝됐고, 타 하드웨어 포팅은 몇 달~몇 년 뒤처진다.

진짜 해자는 두 가지라고 본다. 하나는 이 스택 대부분을 NVIDIA가 월급 주지 않는 사람들이 쓴다는 것 — 20년치 서드파티 kernel·라이브러리·툴링과 API를 배운 수백만 개발자다. 다른 하나는 NVIDIA가 실리콘과 함께 사람을 보낸다는 것이다. 자사 엔지니어 수십 명이 frontier lab과 hyperscaler 팀 안에 상주하며 새 모델 아키텍처마다 kernel을 쓰고 새 실리콘 세대마다 튜닝한다. NVIDIA를 떠난다는 건 kernel을 다시 쓰는 게 아니라 엔지니어링 조직 전체의 멘탈 모델을 재교육하고, 지금 건물 안에 앉아 있는 NVIDIA 엔지니어를 잃는 일이다.

NVIDIA가 거는 베팅을 정리하면 이렇다.

- **Programmability** — 워크로드는 움직이는 표적(attention 변종, 새 모델 구조)이니 모든 블록을 프로그래머블하게 두고 CUDA를 쓰게 한다.
- **Hide latency with massive multithreading** — 레이턴시는 예측 불가·데이터 의존적이니 정적 스케줄이 아니라 스레드 과잉 할당(SM당 최대 64 resident warp)으로 숨긴다.
- **Warp-wrapped matmul** — 행렬 유닛을 fixed-function pipe로 노출하지 않고 warp/thread 추상화 뒤에 둔다. 덕분에 한 kernel이 matmul·softmax·element-wise를 한 번에 fuse한다.
- **Async memory hierarchy** — L2는 유지하되 SMEM·TMEM을 이름 있는 scratchpad로 노출하고 TMA·mbarrier를 얹는다. 컴파일러의 정적 스케줄이 아니라 kernel 안의 software pipelining이다.
- **Amortised SIMT tax** — warp scheduler·레지스터 파일·coherent cache에 쓴 트랜지스터는 MAC에 못 쓴 트랜지스터다. 세금을 인정하되 Tensor Core를 키워 분모를 늘리고 TMEM 같은 유닛으로 MAC 밀도를 산다.
## 3. Google TPU — matrix multiplication machine

TPU의 철학은 정반대에서 출발한다. 아무 병렬 워크로드나 돌리는 프로그래머블 칩 대신 단 하나의 primitive(큰 systolic array 위의 dense matmul)에 집중하고, XLA 컴파일러가 모든 사이클과 모든 바이트를 사전에 계획한다. hardware scheduler 없음, cache 없음, thread/warp 없음. 그래픽도 과학계산도 하지 않는다 — Google의 워크로드(search, translation, recommendation, Gemini)를 와트당 가장 싸게 돌리기 위해 존재한다.

### 계보

| 연도 | 세대 | 핵심 |
| --- | --- | --- |
| 2015 | v1 | 최초 양산 딥러닝 ASIC, INT8 추론 전용 |
| 2017 | v2 | 첫 학습 지원, MXU를 INT8→BF16으로, dual TensorCore + HBM |
| 2020 | v4 | 첫 재구성 가능 OCS(Palomar), SparseCore, 4,096칩 pod |
| 2023 | v5e / v5p | 효율형 / 성능형 분화, 8,960칩 pod |
| 2024 | Trillium (v6e) | 첫 256×256 MXU, Gemini 2.0 학습 |
| 2025 | Ironwood (v7) | reasoning 추론용, native FP8, 9,216칩 superpod = 42.5 ExaFLOPS FP8 |
| 2026 | v8t / v8i | 학습형/추론형, native FP4, 9,600칩 superpod = 121 ExaFLOPS FP4 |

### TensorCore 구성

연산 단위는 TensorCore(flagship은 패키지당 2개, 효율형 v4i·v5e·v6e는 1개)다. 내부는 다섯 컴포넌트로 이뤄진다 — MXU(행렬), VPU(element-wise), Scalar Unit(지휘), XLU(cross-lane reduction), Transpose/Permute Unit. 전부 하나의 VLIW issue plane 위에 있고, Core Sequencer가 매 사이클 322-bit bundle의 8개 슬롯을 전부 채운다. instruction cache miss도, warp scheduler도, out-of-order도, branch predictor도 없다. 컴파일러가 스케줄러이고, 아낀 실리콘 면적은 전부 MAC에 쓴다.

MXU는 weight-stationary systolic array다. v1은 256×256 INT8 1개였고, v2가 128×128(BF16 곱 + FP32 누산)을 도입했는데 TensorCore당 개수가 1→2→4로 늘었다. Trillium부터 다시 256×256(배열당 사이클마다 65,536 MAC)으로 회귀했고 Ironwood·8t·8i도 유지한다.

동작 방식은 이렇다. 행렬 B의 값을 cell마다 하나씩 미리 적재(weight-stationary)하고, activation이 왼쪽 가장자리로 들어와 사이클마다 한 칸씩 전파하며 각 cell의 상주 weight와 곱해지고, partial sum이 아래로 흘러 accumulator queue로 떨어진다. 배열에 들어간 뒤로는 메모리 접근이 없다 — weight는 지나가는 모든 activation에 재사용되고, activation은 row를 따라 128(또는 256)번 재사용된다. 데이터 재사용이 cache의 중재가 아니라 배선에 박혀 있는 것이다.

이게 왜 중요하냐면, 연산의 지배적 비용은 곱셈(수 pJ)이 아니라 메모리 읽기/쓰기(접근당 100~1000배 에너지)이기 때문이다. systolic array는 그 비용을 구조적으로 삭제한다. 대가는 underfill이다 — 256×256 배열에서 128×128 matmul을 돌리면 실리콘 75%가 논다. 그래서 XLA가 차원을 128(v6e+는 256)의 배수로 tiling·padding하고, 모델 코드도 그 quantum을 의식하며 짠다.

VPU는 1D SIMD가 아니라 2D vector machine이다. VREG의 shape은 v4/v5p 기준 (8, 128)이다 — lane 128개 × sublane 8개, (lane, sublane)마다 독립 FP ALU 4개. lane 축이 systolic array 입력 폭과 맞춰져 있다. 현대 TPU 프로그램 속도의 대부분은 VPU/MXU overlap에서 나온다 — quantisation, layernorm, softmax, activation, bias-add가 뒤에서 MXU가 matmul 돌리는 바로 그 사이클에 VPU에서 돌아간다. cross-lane reduction은 XLU가 맡는데 느리고 비싸며 컴파일러의 알려진 hot spot이다.

Scalar Unit은 가장 작지만 가장 결정적인 블록이다. 단일 스레드 dual-issue 정수 ALU, 32비트 레지스터 32개, control state용 SMEM 4 KiB로 구성된다. instruction fetch를 하는 유일한 블록으로, 매 사이클 322-bit bundle을 당겨서 자기 슬롯 2개(주소 연산·loop counter·branch·sync 체크)를 처리하고 나머지 6개를 흩뿌린다 — vector ALU 2, vector load/store(HBM↔VMEM DMA) 2, matrix(MXU queue push/pop) 2. 블록 간 동기화는 sync flag로 명시적이고, 의존성 추적은 하드웨어가 아니라 컴파일러가 barrier check를 삽입해 처리한다.

### 메모리 — cache가 아예 없다

- **HBM**: v2/v5e 16 GB, v3/v4/v6e 32 GB, v5p 95 GB, Ironwood 192 GB, v8 세대 216~288 GB
- **VMEM**: VPU와 MXU 입력 큐를 먹이는 vector scratchpad. v4 32 MiB → v5e 128 MiB → v8i는 384 MiB(KV cache 전체를 온칩에 두려는 의도)
- **CMEM**: v4에서 도입, 128 MiB. HBM과 VMEM 사이의 느리고 큰 SRAM 스테이징
- **SMEM**: Scalar Unit 전용 control state(v4 기준 약 10 MiB)

모든 tensor는 컴파일 시점에 한 tier에 고정되고, XLA의 buffer-assignment pass가 "소비되는 사이클 직전에 도착하도록" DMA를 스케줄한다. 하드웨어는 prefetch도, eviction도, coherence도 하지 않는다. 컴파일러가 맞히면 배열은 절대 멈추지 않고, 틀리면 fallback 경로가 없다.

추천·랭킹 모델은 embedding lookup(거대한 테이블에 수십억 인덱스)으로 산다. 접근 패턴이 dense matmul의 정반대다 — 불규칙, 간접, all-to-all. 256×256 systolic array는 정확히 틀린 모양이다. 그래서 나온 게 SparseCore다. compute tile 16개와 전용 SPMEM을 가진 dataflow processor로 TensorCore 옆에 붙어 scatter/gather/segmented-reduce와 sharded embedding table이 만드는 데이터 의존적 all-to-all을 흡수한다. die 면적·전력의 약 5%로 embedding 중심 모델에서 5~7배 speedup을 낸다. v8i(Zebrafish)는 SparseCore를 아예 빼고 I/O chiplet에 CAE(Collectives Acceleration Engine)를 넣었는데, 문제는 다르지만(autoregressive decode 중 collective reduction) 아이디어는 같다.

### Numerics

v1 INT8 → v2 BF16(FP32와 같은 dynamic range, 메모리 절반, loss scaling 불필요) → v4 native INT8 복귀 → Ironwood native FP8 → v8 native FP4 + MXU 내부 block-scale multiplication(Ironwood가 아직 내던 VPU dequant 오버헤드를 삭제)으로 이어진다. 모든 현대 TensorCore가 stochastic rounding을 하드웨어로 지원하는데, 하위 mantissa 비트를 확률로 써서 반올림하면 긴 학습에서 저정밀 누산의 기댓값이 보존된다.

### Scaling — NVIDIA의 정반대

NVLink+NVSwitch가 다른 GPU의 HBM을 로컬 메모리처럼 보이게 만드는 hardware-managed coherent 주소 공간이라면, Google의 ICI는 message-passing이다. remote-load 시맨틱도, cache coherence도, crossbar도 없다. 모든 multi-chip 연산은 XLA가 컴파일한 명시적 collective다. scale-up domain은 switch fabric이 아니라 torus(이웃끼리 직결 + edge wrap)로 엮이고, rack 경계에서 optical circuit switch가 꿰맨다.

ICI 링크는 TPU die에서 바로 나오는 고속 serial lane이다. 64칩 cube(4×4×4, 액랭 rack 하나) 안에서는 direct-attach copper, cube 사이는 광이다. 칩당 aggregate ICI 대역폭은 v2 약 250 GB/s → Ironwood 1.2 TB/s 양방향 → v8t는 그 2배로 늘었다. 토폴로지는 세대별로 번갈아 간다 — 효율형은 2D torus(v2/v3/v5e/v6e), flagship은 3D torus(v4/v5p/v7/v8t)다.

NVIDIA에 대응물이 없는 부품이 하나 있는데, Palomar OCS다. 3D-MEMS optical circuit switch로, 작은 거울이 물리적으로 회전해 입력 fiber를 임의의 출력에 매핑한다. v4 superpod은 Palomar 48대로 cube 64개(4,096칩)를 하나의 3D torus로 엮는다. 재구성은 ns가 아니라 ms급이지만 상관없다 — circuit-switched이기 때문이다. job 시작 시 토폴로지를 고르고 일주일 돌린 뒤 다음 워크로드에 맞게 재구성한다. 이 한 부품이 세 문제를 동시에 푼다. 워크로드별 토폴로지 재구성(twisted torus로 bisection 최대 70% 개선), on-demand sub-pod slicing, 그리고 fault tolerance다 — 칩이 죽으면 OCS가 예비 cube를 광학적으로 끼워 넣고 ICI domain을 잃지 않은 채 run이 계속된다.

그래서 scale-up의 단위는 superpod이다 — 역할은 NVL72와 같지만 규모가 두 자릿수 크다. v4 4,096칩 → v5p 8,960 → Ironwood 9,216칩(64칩 cube 144개, HBM 1.77 PB ~68 PB/s, 42.5 ExaFLOPS FP8을 하나의 coherent ICI domain으로)까지 왔다. v8t(Sunfish)는 9,600칩 / HBM 2 PB / 121 ExaFLOPS FP4다.

v8i(Zebrafish)는 1,024칩 / HBM 약 295 TB / 약 10 ExaFLOPS FP4인데, torus를 버리고 Boardfly라는 계층형 high-radix 토폴로지(4칩 ring → 8보드 group → OCS로 최대 36 group)를 쓴다. 이유가 흥미롭다 — 3D torus는 collective가 nearest-neighbour일 때 최강이지만(ring all-reduce는 매 사이클 모든 링크를 쓴다), MoE expert routing은 정반대인 all-to-all이라 왕복 레이턴시가 최장 hop 쌍에 묶인다. 1,024칩 3D torus의 diameter가 16 hop인데 Boardfly는 7 hop으로 압축했다.

v7까지는 단일 fabric인 Jupiter를 썼다(2022년부터 spine이 전면 광, Apollo OCS — Palomar와 같은 3D-MEMS 계열을 건물 규모로). 건물당 bisection 13 Pb/s다. rack에서 datacenter spine까지 모든 레이어에 같은 primitive(OCS)를 쓰는 건 Google만의 시그니처다.

v8t에서 scale-out이 둘로 쪼개진다. east-west TPU↔TPU 트래픽은 Virgo라는 전용 accelerator fabric으로, Jupiter는 north-south(스토리지 접근, 범용 컴퓨트, 사이트 간)를 맡는다. Virgo는 high-radix switch로 만든 flat, two-layer, non-blocking 토폴로지다 — 어떤 TPU든 다른 TPU까지 최대 switch 2홉. 클러스터 하나가 TPU 8t 134,000개 이상을 bisection 47 Pb/s로 연결하고(칩당 대역 4배, unloaded 레이턴시 40% 감소), multi-planar fault isolation과 sub-millisecond telemetry로 스케줄러가 straggler를 스텝 망치기 전에 죽인다. 아키텍처적 보상은 레이어별 독립 진화다 — scale-up, east-west scale-out, front-end가 서로 재배선 없이 각자 주기로 간다.

칩당 scale-out 대역은 Ironwood 100 Gbps 수준, v8t는 4배지만 여전히 칩당 ICI보다 두 자릿수 낮다. 이 격차가 partitioning을 규정한다 — tensor parallelism과 MoE expert routing은 ICI 안에, data/pipeline parallelism은 scale-out으로.

위에는 Multislice(XLA에 배선되어 단일 SPMD 프로그램이 여러 pod의 slice를 가로지름, 계층형 collective 생성)와 Pathways가 있다. NCCL+Slurm+Megatron 계열이 여러 controller에서 SPMD를 구동한다면, Pathways는 단 하나의 client에서 job 전체를 구동하고 여러 "island"(각자 ICI domain을 가진 pod)를 가상화한다. gang scheduling, elastic training(slice 장애 시 OCS가 토폴로지를 재구성하고 Pathways가 새 shape에서 마지막 체크포인트로 재개), cross-region orchestration까지 다룬다. Gemini Ultra는 여러 데이터센터에 걸쳐 학습된 최초의 frontier 모델이고, Pathways가 그걸 하나의 동기 SPMD job으로 꿰맸다.

### Software — compiler-driven vs kernel-driven

GPU에서는 개발자가 kernel을 쓰고 프레임워크가 kernel을 엮으며 컴파일러 역할은 대체로 국소적이다. TPU에서는 개발자가 JAX로 수치 프로그램을 쓰고 그 아래 전부를 XLA가 책임진다 — 어떤 연산을 fuse할지, 각 tensor가 어디 살지, 2D vector register에 어떻게 배치할지, HBM→VMEM DMA를 언제 쏠지, 322-bit VLIW bundle을 어떻게 스케줄할지, 수천 칩에 어떻게 shard할지까지. 나쁜 스케줄을 덮어줄 하드웨어 fallback이 없다.

컴파일 경로는 `JAX → JAXpr → StableHLO → HLO → LLO → VLIW bundle`이다. XLA pass pipeline은 operation fusion(중간값이 HBM에 안 닿게), layout assignment(레지스터와 systolic 입력이 둘 다 2D라 1D SIMD보다 훨씬 어렵다), buffer assignment(VMEM/CMEM/HBM 중 하나에 pin), SPMD partitioning, 그리고 8슬롯을 전부 채우는 VLIW scheduler로 이어진다.

multi-chip 실행은 SPMD로 하는데, GSPMD(2026 초 MLIR 네이티브 후속 Shardy로 교체 중)가 생성한다. 사용자는 Mesh + PartitionSpec으로 핵심 tensor 몇 개에만 선언적으로 sharding을 달고, 컴파일러가 그래프 전체로 전파하며 layout이 바뀌는 지점에 all-reduce/all-gather/reduce-scatter를 끼워 넣는다. PyTorch 관용구(FSDP/DeepSpeed가 런타임으로 module 경계에서 collective를 발사)의 정확한 반대다.

Pallas가 탈출구다 — JAX의 kernel 작성 언어로, GPU의 Triton에 대응한다. Mosaic(MLIR 기반 TPU 백엔드)을 거쳐 LLO로 내려가고 custom op으로 HLO에 다시 박힌다. 존재 이유는 XLA가 새로운 attention 변종, fused MoE dispatch, 수동 VMEM tiling과 DMA 스케줄링이 필요한 것들의 최적해를 늘 합성해내지는 못하기 때문이다 — 즉 FlashAttention급 최적화가 필요할 때 쓴다.

PyTorch 경로는 실재하지만 2등 시민이다. torch_xla의 LazyTensor(모든 op을 HLO 그래프에 기록, barrier에서 컴파일, graph-shape 해시로 캐시)에서 시작해 GSPMD 스타일 sharding annotation, `torch.compile` 통합, JAX bridge가 붙었다. 그래도 격차가 실재해서 vLLM TPU(tpu-inference 플러그인)는 JAX 모델이든 PyTorch 모델이든 전부 단일 JAX→XLA 경로로 내린다. 2026년 4월 발표된 TorchTPU가 Google의 답이다 — eager mode, `torch.distributed`, `torch.compile`을 갖춘 네이티브 PyTorch 경험을 노린다.

CUDA와 비교하면 TPU 생태계는 산재가 아니라 집중이다. 프레임워크 아래 거의 전부(XLA, JAX, Flax, Optax, Pallas, MaxText, Pathways, Shardy, Mosaic)를 Google이 직접 오픈소스로 내고 실리콘과 보조를 맞춰 진화시킨다. 서드파티 kernel은 CUDA의 수십 년 축적보다 훨씬 적다 — 워크로드가 이상할수록 해자가 얇고, 워크로드가 Gemini를 닮을수록 깊다. Triton과 `torch.compile`이 NVIDIA 쪽에서 격차를 좁히며 kernel-driven과 compiler-driven이 수렴 중이지만, 철학적 양극은 여전히 실재한다 — TPU에서 컴파일러는 유일한 인터페이스고, GPU에서 컴파일러는 여러 인터페이스 중 하나다.

TPU가 거는 베팅을 정리하면 이렇다.

- **Systolic array** — matmul이 워크로드를 지배하니 실리콘을 systolic array에 쓴다.
- **Software scratchpad** — 연산은 싸고 메모리는 비싸니 배열의 배선 안에서 데이터를 재사용하고 cache를 software-managed scratchpad로 대체한다.
- **Compiler scheduling** — 워크로드가 정적으로 예측 가능하니 스케줄링을 컴파일러로 옮긴다 — VLIW issue, speculation·OoO·동적 스케줄러 전부 삭제한다.
- **MAC-only silicon** — peak보다 전력이 중요하니 곱셈-누산을 하지 않는 트랜지스터는 전부 지운다 — cache tag, branch predictor, reorder buffer가 대상이다.
- **Dedicated off-array engine** — dense matmul 배열이 틀린 모양인 실제 워크로드(embedding, collective)는 메인 코어를 비틀지 말고 전용 소형 엔진(SparseCore, CAE)을 옆에 깎아낸다.

## 4. AMD Instinct — CU는 보수적으로, 패키지에 재투자

NVIDIA가 세대마다 SM이 할 수 있는 일을 늘린다면, AMD는 2012년 GCN 이후 Compute Unit을 보수적으로 유지하고 패키지에 재투자했다. 2021년 이후 매 세대 동시대 NVIDIA flagship과 HBM 용량이 동급 이상이고, 최초의 3D-stacked 데이터센터 GPU(CDNA 3), 최초의 coherent CPU+GPU APU(MI300A), 그리고 개방 생태계(ROCm, HIP, OCP MX, UALink)로 승부를 본다.

### 용어 대응

| AMD | NVIDIA |
| --- | --- |
| Compute Unit (CU) | Streaming Multiprocessor (SM) |
| SIMD | SM Sub-Partition |
| SIMD Lane | CUDA Core (FP32 ALU) |
| Wavefront (wave64) | Warp (warp32) |
| Matrix Core | Tensor Core |
| MFMA | mma.sync / wgmma / tcgen05.mma |
| LDS (Local Data Share) | SMEM (Shared Memory) |
| Infinity Fabric | NVLink |

### 연산 — MFMA는 제자리에 머물렀다

CU 자체는 보수적이다. 16-lane SIMD 4개, 공유 scalar unit, LDS, L1 vector cache, SIMD별 VGPR + CU 공유 SGPR, 그리고 CDNA 1부터 MFMA를 돌리는 Matrix Core로 구성된다. 모양은 2012년 GCN 이후 실질적으로 안 변했고, 변하는 건 개수(MI100 120 → MI250X 220 → MI300X 304 → MI355X 256)와 그걸 묶는 패키징이다.

핵심 대조가 여기서 나온다. NVIDIA의 Tensor Core는 스레드 계층을 올라갔다 — Volta 32스레드 warp → Hopper 128스레드 warp-group → Blackwell 단일 스레드 + 선택적 2-SM cluster. AMD의 Matrix Core는 그대로 머물렀다. MI100(2020)부터 MI355X(2025)까지 모든 MFMA 세대가 wavefront-scoped다 — wave64 하나가 matrix op(`V_MFMA_*`)을 issue하고 SIMD 4개가 협력해 구동하며, operand는 wavefront의 레지스터 파일(A·B는 VGPR, C·D는 보통 전용 AGPR)에서 온다. 명령은 빨라지고 포맷은 넓어졌지만 issuer와 scope는 바뀌지 않았다.

이 선택에는 두 가지 비용이 붙는다. 하나는 divergence다 — 반만 찬 wave64는 lane 32개를 버리고, 반만 찬 warp32는 16개를 버린다. control flow가 대체로 균일하면 싼 값이지만 불규칙한 워크로드에서는 아프다. 다른 하나는 overlap이다 — NVIDIA의 비동기 descriptor 기반 matmul은 issue와 실행을 분리하는데, AMD의 wavefront-collective MFMA에는 대응물이 없다. matmul을 issue한 바로 그 wave는 그게 pending인 동안 의미 있는 vector 작업을 못 한다. 별도의 wavefront 간 overlap은 가능하지만 명시적 wavefront barrier로 소프트웨어에서 무대를 짜야 해서 더 깨지기 쉽고 wave slot·레지스터를 더 먹는다.

이게 얼마나 아픈지는 워크로드에 달렸다. 순수 dense GEMM(대배치 학습의 inner loop)에서는 matmul 중에 할 유용한 일이 없고 두 엔진 다 포화라 async의 이득이 적다 — 실제로 AMD가 exascale HPC에서 앞섰던 영역이다(Frontier의 MI250X, El Capitan의 MI300A). Transformer attention(FA3, FA4)에서는 matmul과 softmax·masking·KV cache 읽기가 교차하고 async overlap이 그 kernel의 구조 그 자체라, AMD는 파이프라인을 손으로 재현해야 하고 이게 뒤처지는 요인이다. MoE dispatch, paged attention, speculative decode도 같은 진영이다 — matmul 옆에서 같이 돌고 싶어하는 주소 불규칙 작업이기 때문이다.

| 세대 | 추가/변화 |
| --- | --- |
| CDNA 1 (2020) | FP32 256 / FP16 1024 / BF16 512 / INT8 1024, A100과 나란히 native BF16 |
| CDNA 2 (2021) | FP64를 full-rate matrix로 배증(256) — AMD만의 베팅, MI250X를 Frontier에 올린 판단 |
| CDNA 3 (2023) | FP8 4,096으로 H100과 동급, 2:4 structured sparsity, TF32 상당 경로 |
| CDNA 4 (2025) | FP4 16,384 + FP6 (OCP MX block scaling), 한 MFMA 안에서 A/B 정밀도 혼합(FP8 × FP4) 가능. 동시에 CU당 FP64를 절반으로 |

CDNA 4의 변곡점은 따로 적을 가치가 있다 — MI300X는 학습·HPC·추론을 함께 섬겼지만 MI355X는 AI 칩 우선이다. Frontier를 돌린 full-rate FP64-matrix 베팅이 죽은 건 아니지만 더 이상 무게를 지지 않는다.

### 메모리 — 층은 적고, NVIDIA에 없는 거대 cache가 하나

CU에서 바깥으로 나가면 LDS 64 KB(software-managed 32-bank) → vector L1(16 KB → MI300X부터 32 KB) → XCD별 L2 수 MB 순이다. 그런데 L2는 XCD 간 coherent하지 않고, coherence는 한 층 위에서 일어난다.

그 층이 Infinity Cache다 — MI300X 기준 256 MB, IOD 4개에 분산, 16-way set-associative, 실측 약 12 TB/s로 MI300X HBM3의 5.3 TB/s보다 두 배 이상이다. 원래 좁은 GDDR 버스를 보상하려고 RDNA 게이밍 GPU에서 나온 IP를 CDNA 3에서 AI로 재사용했는데, attention KV 재사용과 weight 재사용이 큰 LLC에 유난히 잘 맞았다. NVIDIA는 더 큰 HBM 대역폭에(B200 8 TB/s, Rubin은 HBM4) 걸었고, AMD는 cache에 걸었다.

HBM 용량은 공격적으로 키웠다 — MI100 32 → MI210 64 → MI250X 128 → MI300X 192 → MI325X 256 → MI350X 288 GB. 2021년 이후 매 세대 동시대 NVIDIA flagship 이상이다. 베팅은 "추론 워크로드는 점점 capacity-bound가 되고, 메모리 많은 칩이 이긴다"는 것이다.

### Chiplet — CDNA가 NVIDIA와 갈라지는 지점

CDNA 1 MI100은 monolithic 7 nm였다. CDNA 2 MI250X는 Aldebaran GCD 2개를 2.5D EFB 기판에 나란히 놓고 in-package Infinity Fabric 4링크(합 400 GB/s)로 연결했지만 소프트웨어에는 GPU 2개로 보였다.

CDNA 3가 판을 바꿨다. XCD 8개(TSMC N5, 각 약 115 mm²)를 TSMC SoIC hybrid bonding으로(서브마이크론 피치 TSV, microbump 없음) 아래쪽 I/O die 4개(TSMC N6) 위에 3D로 쌓는다. IOD가 Infinity Cache, HBM3 PHY, Infinity Fabric, PCIe Gen 5를 지고, IOD 하나당 위에 XCD 2개·옆에 HBM 2스택이 붙는다. IOD 4개는 Infinity Fabric AP로 bisection 4.8 TB/s에 엮여서, 1,530억 트랜지스터 패키지가 kernel에는 GPU 하나로 보인다. NVIDIA는 H100까지 monolithic이었고 B200에서야 2.5D CoWoS-L로 reticle 한계 die 2개로 갔다 — AMD가 한 세대 먼저, 더 작은 die 면적으로 3D 적층에 도달한 셈이다.

MI300A APU는 베팅을 더 밀었다. XCD 8개 중 2개를 Zen 4 CCD 3개로 교체하고 HBM·Infinity Cache·IOD는 그대로 둔 채 CPU와 GPU가 HBM3 기반 하나의 물리 주소 공간을 하드웨어 coherence로 공유한다. host-device copy 없음, pinned memory 없음, 경로에 PCIe 없음. Zen 4 코어와 CDNA 3 XCD가 같은 페이지를 읽는다. NVIDIA의 Grace-Hopper가 두 패키지를 NVLink-C2C로 잇는다면 MI300A는 하나다. El Capitan(MI300A 4개짜리 노드 11,039개)이 이걸 정당화한 배치다.

MI355X에서는 XCD가 TSMC N3P로 옮겨가며 XCD당 활성 CU가 32개로 줄었다(총 256, MI300X 304 대비 감소 — 더 큰 Matrix Core와 160 KB LDS를 위한 면적 확보). IOD는 4개→2개로 합쳐지고 각자 128 MB씩 Infinity Cache를 진다. IOD 간 Infinity Fabric AP는 bisection 5.5 TB/s, HBM3E 12-Hi 8스택으로 288 GB @ 8 TB/s다. 총 1,850억 트랜지스터.
### Scaling — 2026년까지 rack-scale 답이 없었다

메모리 베팅에는 스케일링 상의 귀결이 있다. MI300X 8장이 HBM 1.5 TB, MI350X 8장이 2.3 TB를 쥐면 405B 모델을 FP8로 8-GPU 박스 하나에 통째로 올릴 수 있다(weight + KV cache + long context·큰 배치 여유까지). 같은 모델이 8× H100(640 GB)에서는 정교한 sharding을 요구한다. 2024~2025 추론 워크로드에서는 AMD의 scale-up이 rack에서 NVL72를 이기지 않아도 박스에서 경쟁력이 있었다. 다만 frontier 학습에서는 그게 필요했고, AMD에겐 2026년까지 답이 없었다.

MI355X까지 AMD의 scale-up은 Infinity Fabric 위의 8-GPU OAM 플랫폼을 뜻했다. MI300X는 IF 링크 7개(박스 내 모든 peer에 하나씩) × 128 GB/s 양방향 = GPU당 896 GB/s mesh, fully-connected all-to-all이다. MI350X는 링크당 153.6 GB/s(GPU당 약 1,075 GB/s)로 올렸지만 8-GPU 형태는 유지했다. 플랫폼은 OCP UBB 2.0을 따르므로 NVIDIA HGX baseboard와 기계적 소켓이 같다 — 서버 벤더가 같은 섀시에 AMD든 NVIDIA든 실을 수 있다.

Helios가 rack-scale 공백을 메우러 온다(2026년 하반기, MI455X와 함께). rack당 72 GPU, HBM4 약 31 TB, HBM 총대역 1.4 PB/s, 2.9 ExaFLOPS FP4 / 1.4 ExaFLOPS FP8, scale-up 260 TB/s, scale-out 43 TB/s 규모다. 폼팩터는 AMD 독자 섀시가 아니라 Open Rack Wide(ORW) — Meta가 2025년 OCP에 제출한 더블와이드 액랭 규격이다. ORW로 표준화한 hyperscaler라면 별도 데이터센터 설비 공사 없이 Helios를 배치할 수 있게 하려는 의도적 베팅이다.

fabric은 UALink(Ultra Accelerator Link)다 — AMD가 Apple, AWS, Cisco, Google, HPE, Intel, Meta, Microsoft, Synopsys와 함께 만든 개방 컨소시엄 표준이다. UALink 200G 1.0(2025-04)은 200 GT/s lane, 방향당 800 Gbps, switched 토폴로지로 pod당 1,024 accelerator까지 지원한다. 약속은 "NVLink에 비견되는 cache-coherent interconnect인데 누구의 소유도 아닌 것"이다.

다만 함정이 있다. native UALink switching 실리콘이 2027년까지 양산되지 않는다. Astera Labs의 Scorpio와 Auradine·Enfabrica·Xconn의 경쟁 부품 모두 2026년 말~2027년 배치를 노린다. 그래서 Helios는 출시 시점에 UALoE(Infinity Fabric을 표준 Ethernet에 터널링)를 임시방편으로 쓴다 — 프로그래밍 모델은 보존하면서 native UALink fabric을 기다리는 것이다. native UALink switching은 2027년 MI500과 함께 온다. 즉 출시 시점의 Helios는 NVL72의 진짜 cache-coherent NVLink domain보다는 빠른 Ethernet 터널 기반 coherent 클러스터에 가깝다. 2026년 하반기에 경쟁력 있는 제품을 내놓는 대가로 지불한 실질적 타협인 셈이다.

AMD는 InfiniBand를 팔지 않는다. scale-out 스택 전체가 Ethernet이고, 또 다른 개방 표준인 UEC(Ultra Ethernet Consortium)에 정박해 있다.

UEC 1.0(2025-06)이 정의하는 UET(Ultra Ethernet Transport)는 표준 Ethernet 위의 새 RDMA transport로, packet spraying, SACK 기반 selective retransmission, 현대적 congestion control을 갖는다. UET은 RoCEv2가 아니다 — RoCEv2가 InfiniBand transport를 Ethernet 프레임에 캡슐화한 것이라면, UET은 scale-out AI fabric을 위해 RDMA 시맨틱을 처음부터 다시 설계한 것이다. AMD는 Broadcom, Cisco, Meta, Microsoft와 함께 창립 멤버다 — UALink와 같은 수, "구현이 아니라 표준을 소유한다"는 전략이다.

NIC는 2022년 인수한 Pensando가 맡는다. Pollara 400(400 GbE, P4 프로그래머블, UEC-ready, PCIe Gen 5), 2026년 MI455X와 함께 나올 Vulcano 800(UEC 1.0 준수, PCIe Gen 6, native UALink 인터페이스, Pollara 대비 GPU당 scale-out 대역 8배), 그리고 프런트엔드 DPU Salina 400이 라인업이다.

다만 switch 실리콘은 AMD 것이 아니다. Helios의 43 TB/s scale-out fabric은 Broadcom Tomahawk 6(102.4 Tbps, CPO "Davisson")를 통과한다. AMD에는 자체 CPO도 자체 switch ASIC도 없다. NVIDIA가 InfiniBand·Spectrum-X·ConnectX·BlueField·Quantum-X Photonics CPO까지 스택 전체를 소유하는 것과 대조적으로, AMD는 한 층(NIC + DPU)만 소유하고 개방 표준 + best-of-breed 파트너 실리콘이 수직통합을 앞지를 것에 건다.

업계는 AMD 쪽으로 움직였다 — Dell'Oro 집계로 2025년 AI scale-out fabric 물량에서 Ethernet이 InfiniBand의 2배 이상이다. AWS, Microsoft, Meta, Oracle, xAI 모두 자사 AMD 기반 AI 클러스터에 Ethernet으로 표준화했다. 남은 질문은 "Ethernet이 RDMA 시맨틱에서 IB를 따라잡을 수 있는가"(UEC가 그 격차를 닫았다)가 아니라, Helios가 NVL72와의 rack-scale 격차를 frontier 학습 물량을 가져올 만큼 빨리 닫을 수 있는가다.

### Software — ROCm

NVIDIA 스택이 독점적·수직통합형(cuBLAS·cuDNN·TensorRT-LLM이 NVIDIA 혼자 유지하는 바이너리 blob)이라면, ROCm은 GitHub 네이티브이며 walled-garden 라이브러리 세트 대신 개방 표준(PyTorch, Triton, vLLM, OCP MX)에 건다.

HIP은 CUDA 호환 C++ 런타임으로, `hipify`가 CUDA 소스를 자동 변환한다. HPC 대량 코드(HACC, Laghos, QMCPack)는 80~95%가 그대로 포팅된다(CORAL-2 수치). 다만 현대 AI kernel은 훨씬 나쁘다 — Hopper/Blackwell 고유 primitive(TMA descriptor, `wgmma`, `tcgen05.mma`)에 손 대는 코드는 깔끔한 ROCm 대응물이 없어 손으로 다시 써야 한다.

라이브러리 티어는 이름까지 1:1 대응한다 — rocBLAS↔cuBLAS, hipBLASLt↔cuBLASLt, MIOpen↔cuDNN, RCCL↔NCCL, Composable Kernel(및 ck-tile DSL)↔CUTLASS, rocprof 계열↔Nsight. 다만 TensorRT-LLM의 1st-party 대응물은 없다. 대신 vLLM을 오픈소스 서빙 엔진으로 밀고 AMD 전용 operator(AITER)를 꽂는다 — vLLM 전용 ROCm CI가 2026년 초 테스트 통과율을 37%→93%로 끌어올렸다.

PyTorch 경로는 1급이다. eager mode는 2018년부터, `torch.compile`은 Triton으로 내려가며 Triton의 ROCm 백엔드(+AOTriton)는 upstream이다. XLA 같은 중간 IR은 없고 HIP/Triton/CK로 직접 컴파일한다. Triton이 PyTorch의 기본 kernel 경로가 될수록 포팅 비용의 상당 부분이 증발하는데, AMD 개방 전략 밑의 아키텍처적 베팅은 "Triton의 Python DSL이 cross-vendor lingua franca가 되어 CUDA급 kernel 생태계를 따로 만들 필요를 우회한다"는 것이다.

FlashAttention이 하중을 받는 사례다. FA2는 Composable Kernel로 MI300X에서 프로덕션이다. FA3(Hopper 튜닝)는 AITER+CK로 부분 지원이지만 Dao-AILab 정본은 CUDA 전용이다. FA4(Blackwell, 2026-03)는 ROCm 포트가 아예 없다. Hazy Research의 HipKittens(ThunderKittens의 MI355X 포트, 2025-11)는 약 500줄로 손튜닝 AITER와 forward-pass 동급을 주장한다. 패턴은 이렇다 — 오픈소스 학계 kernel이 AMD의 꼬리를 수년이 아니라 수개월 뒤에 닫는다.

배치는 전략을 검증했다 — Azure ND MI300X v5(2024-05 GA)에서 OpenAI가 GPT 추론을 돌리고, Meta는 Grand Teton 플랫폼으로 Llama 3/4 추론을, Oracle OCI도 GA했다. 파일럿이 아니라 실제 서빙 fleet이다.

솔직한 격차도 있다. 독립 벤치마크(Phoronix, 2026-03)에서 ROCm 7.2는 동일 실리콘·동일 정밀도의 표준 PyTorch/vLLM/SGLang 워크로드에서 CUDA 대비 10~25% 느리다. ROCm 7은 feature parity에 도달했지만 perf parity는 아니다. FA4 꼬리(Blackwell 최신 primitive를 파먹는 연구 코드)가 NVIDIA 해자가 가장 견고한 지점이다. NVIDIA는 frontier lab 안으로 엔지니어를 보내고, AMD는 GitHub로 kernel을 보낸다는 대비가 여기서 나온다. 흔한 워크로드(Llama 추론, attention, dense transformer 학습)에서는 두 전략이 수렴하지만, 새로운 연구 코드의 long tail은 여전히 MI300X/MI355X 배치에 NVIDIA 사용자는 안 내는 엔지니어링 시간을 물린다.

AMD가 거는 베팅을 정리하면 이렇다.

- **HPC 먼저, 그다음 AI** — CDNA 2~3까지 full-rate FP64 matrix를 싣고, 추론 경제성이 저정밀로 확실히 기운 CDNA 4에서 분기한다.
- **메모리 용량** — 2021년 이후 매 세대 HBM 용량 동급 이상 + H100이 HBM까지 가야 하는 재사용을 흡수하는 256 MB Infinity Cache.
- **조기 3D 적층** — NVIDIA보다 먼저 연산을 cache·I/O 위에 3D로 쌓는다(2023 SoIC vs NVIDIA 2025).
- **Coherent CPU+GPU** — MI300A APU는 출하된 제품 중 가장 chiplet-공격적이고, El Capitan이 증거다.
- **개방 scale-up fabric** — NVLink와 독점 FP4 대신 UALink와 OCP MX.

## 5. Cerebras WSE — 웨이퍼를 자르지 않는다

memory wall은 웨이퍼를 잘랐기 때문에 생긴 결과라는 게 Cerebras의 진단이다. 팹은 300 mm 실리콘에 수십 개 die를 찍고 톱으로 잘라낸 뒤, 업계는 가장 이국적인 엔지니어링(HBM, NVLink, CoWoS, 랙당 구리 5,184가닥)을 쏟아부어 그 조각들을 on-die 대역폭의 극히 일부 수준으로 다시 잇는다. Cerebras는 그 톱질을 건너뛴다.

Wafer-Scale Engine은 말 그대로 실리콘 한 조각이다. reticle field 84개, 46,225 mm², dataflow core 900,000개로 이뤄지고, 온칩 메모리의 모든 바이트가 연산 유닛에서 1 사이클 거리의 SRAM에 있다.

### 계보

| 연도 | 세대 | 핵심 |
| --- | --- | --- |
| 2019 | WSE-1 / CS-1 | 최초 양산 wafer-scale, 1.2T 트랜지스터, 400,000 core, SRAM 18 GB |
| 2021 | WSE-2 / CS-2 | 7 nm, 850,000 core, SRAM 40 GB. weight streaming으로 weight를 MemoryX로 뺌 |
| 2023 | Condor Galaxy | G42와 64-시스템 클러스터, Jais 아랍어 LLM 계열 학습 |
| 2024 | WSE-3 / CS-3 | 5 nm, 4T 트랜지스터, 900,000 core, SRAM 44 GB, core당 FP16 SIMD 8-wide |
| 2024 | Inference 전환 | weight를 streaming 대신 SRAM에 상주 — 업계 최고 실측 decode 속도, 지금의 회사를 정의한 피벗 |

### 아키텍처 — 계층이 아니라 평면

GPU는 계층이다. SM 안의 warp 안의 thread, rack 안의 package 안의 die. 경계마다 고유한 대역폭·레이턴시·프로그래밍 구조가 붙는다. WSE는 평평한 평면이다 — 동일한 core 900,000개가 2D mesh에 가장자리를 맞대고 깔려 있고, 공유 cache도, 글로벌 메모리도, 한 core와 나머지 899,999개 사이의 어떤 경계도 없다.

core 하나는 아주 작다(WSE-2 기준 약 38,000 µm², 절반 SRAM 절반 로직, 피크 30 mW). 로컬 SRAM 48 kB, 범용 레지스터 16개, 6-stage 파이프라인, 4-wide FP16 FMAC SIMD(WSE-3는 8-wide), fabric으로 가는 5-port 라우터로 구성된다. 실행은 dataflow다 — core는 wavelet이 도착할 때까지 놀고, wavelet의 control bit가 어느 handler task를 쏠지 고르며, 하드웨어 microthread 8개가 operand 도착·소진에 따라 사이클 단위로 전환한다. warp도, warp scheduler도, miss날 cache도, reorder buffer도 없다 — 데이터의 도착이 곧 스케줄이다.

스테퍼는 한 번에 reticle 하나(약 850 mm²)를 노광한다. 그래서 모든 일반 칩이 그 천장 아래 산다(그리고 B200은 그 벽에 닿는 순간 2-die가 됐다). Cerebras는 같은 약 550 mm² die를 12×7 격자로 84번 찍고 — 여기까지는 여느 TSMC 고객과 같다 — 그다음 TSMC와 공동 개발한 공정으로 톱이 지나갈 1 mm 미만 scribe line 위에 상위 메탈을 더 깐다. mesh는 이 이음매를 source-synchronous 병렬 인터페이스(WSE-3 기준 die당 2,880 GB/s)로 건너가고, die 간 레이어 전체가 약 97 W를 먹는다. 소프트웨어에게 이음매는 존재하지 않는다 — 균일한 mesh 하나, 칩 하나로 보인다.

1980년대에 wafer-scale이 실패한 이유는 수율이었다 — monolithic 웨이퍼 컴퓨터에 결함 하나면 웨이퍼 전체가 죽는다. Cerebras의 답은 granularity다. H100에서 결함 하나는 약 6 mm² SM 전체를 죽이지만, WSE에서는 0.05 mm² core 하나를 죽인다. WSE-3는 약 970,000 core를 만들어 900,000개를 출하한다 — 약 7% 예비 풀과 중복 fabric 링크로 하드웨어가 모든 결함을 우회 매핑해 완전한 논리 mesh를 복원한다.

이 core에서 특이한 건 datapath가 아니라 명령의 정의다. 범용 레지스터 16개 옆에 44개의 data-structure register(DSR)가 있고, 각각 tensor descriptor(base address, extent, stride, 최대 4차원)를 담는다. 명령은 operand를 DSR로 지칭하므로, FMAC 하나가 "도착하는 스트림을 이 상주 tensor와 곱해서 저기에 누산하라"고 말하고 하드웨어는 tensor가 지속되는 동안 원소를 계속 흘린다. 곱셈을 감싸는 소프트웨어 루프도, 원소당 instruction fetch도 없다 — 루프가 descriptor 안에 산다. NVIDIA가 Tensor Core 다섯 세대에 걸쳐 matmul을 descriptor 기반 단일 커맨드 쪽으로 걸어간 반면, WSE core에서 tensor 명령은 애초에 다른 형태가 없다.

### 연산 — 웨이퍼에 matrix unit이 없다

NVIDIA·Google·AMD는 전부 FLOPs를 전용 matmul 엔진에 집중시키고 그걸 어떻게 먹이는가에서 갈린다. Cerebras는 matmul을 fabric으로 조립한다. GEMM은 웨이퍼 전체의 안무로 돈다 — 도착한 weight가 activation을 쥔 core의 row를 따라 broadcast되고, 각 core가 자기 슬라이스에 대해 multiply-accumulate(weight당 AXPY)를 쏘고, partial sum이 mesh를 가로질러 reduce된다. Tensor Core가 레지스터 타일에서, MXU가 배선에서 얻는 데이터 재사용을 WSE는 기하학에서 얻는다 — activation은 절대 움직이지 않으므로 날아다니는 operand는 지금 곱해지는 것뿐이다.

FLOPs 숫자는 조심해서 읽어야 한다. WSE-3의 헤드라인 125 PFLOPS는 sparse FP16이다 — 이상적으로 희소한 tensor에서 약 8배 zero-skipping 이득을 가정한 값이다. dense는 약 15.8 PFLOPS FP16(900,000 core × 8-wide FMAC × 1.1 GHz로 유도한 값이며, Cerebras는 공식 dense 수치를 내지 않는다). 진짜 compute지만 요점은 아니다 — 와트당 dense FLOPs로는 웨이퍼가 동시대 모든 GPU에 진다. 웨이퍼는 애초에 FLOPs 머신이 아니라 bandwidth 머신이고, FLOPs는 SRAM을 따라가기 위해 존재한다.

zero-skipping이 dataflow가 밥값 하는 지점이다. 연산이 도착한 데이터로 촉발되므로 0은 아무것도 촉발하지 않는다 — 0은 송신 측에서 걸러지고 수신 core는 그걸 보지도, 사이클을 쓰지도 않는다. 이건 비정형·원소 단위 sparsity, 즉 NVIDIA의 2:4 structured sparsity가 표본만 뜨는 일반 케이스다. 그런데 아직 행사되지 않은 옵션이기도 하다 — Cerebras 자체 sparse pretraining 결과는 벤더 저작이고 7B 미만이며, 플래그십 고객 모델 중 sparse 학습으로 공개된 건 없다(최대 규모인 Jais 2도 dense다). 비정형 sparsity를 수확할 수 있는 유일한 실리콘이 아직 그걸 쓰는 대표 모델을 내지 못한 셈이다.

### 메모리 — 계층이 하나다

core 안 48 kB 조각으로 나뉜 SRAM 44 GB, 그리고 웨이퍼 위에 그 외엔 아무것도 없다. HBM 없음, L2 없음, eviction policy 없음. 모든 바이트가 FMAC에서 1 사이클 거리다.

인용되는 21 PB/s는 조심해서 봐야 한다 — 900,000개 로컬 SRAM 포트의 합인 on-wafer aggregate이지 point-to-point 링크가 아니고 HBM 수치와 비교 대상이 아니다. 정직한 비교는 bytes per FLOP이다. 웨이퍼는 dense FP16 FLOP당 약 1.3 바이트를 먹일 수 있고, B200은 HBM에서 약 0.002를 얻는다. 이 축에서 모든 GPU와 TPU는 굶고 있고, WSE만이 균형 잡힌 기계다. 그리고 순수 bandwidth 문제인 decode(토큰당 weight 전체 1회 읽기)야말로 웨이퍼가 맞춰 생긴 국면이다.

섬의 초능력과 감옥은 같은 사실이다. 웨이퍼가 바깥 세상과 연결되는 통로는 12×100 GbE = 1.2 Tb/s로, Blackwell GPU 한 장에 붙는 ConnectX-8 NIC 하나보다 겨우 조금 많다. on-wafer SRAM과 off-wafer Ethernet 사이에 다섯 자릿수(10만 배)가 놓인다. NVIDIA의 계층은 완만하게 내려가지만 WSE는 두 층 사이에 절벽이 있다. 게다가 섬은 자라지 않는다 — 선단 공정에서 SRAM 밀도는 사실상 스케일링을 멈췄다. WSE-3는 노드를 온전히 한 단계 줄이고 트랜지스터를 54% 늘렸는데도 SRAM은 WSE-2보다 10% 많을 뿐이다. 이 아키텍처의 가장 희소한 자원이, 다음 공정 노드가 더 이상 사주지 않는 바로 그것인 셈이다.

### Weight streaming — 흐름을 뒤집는다

GPU/TPU에서는 weight가 상주하고 activation이 흘러간다. WSE에서는 activation이 상주하고 weight가 흘러간다. master weight는 클러스터 옆의 DRAM+flash 어플라이언스인 MemoryX에 산다. 레이어 단위로 weight가 웨이퍼를 가로질러 흐르며 SRAM에 고정된 activation에 대해 MAC을 촉발하고 떠난다. backward pass에서는 gradient가 흘러 나가고, optimizer step은 MemoryX 안의 CPU에서 돈다(weight update는 재사용 없는 O(파라미터)짜리 element-wise 작업이라 CPU급 연산으로 충분하다). 웨이퍼는 weight를 일시적으로조차 저장하지 않는다. 모델 크기는 44 GB가 아니라 MemoryX가 제한하고, 44 GB는 activation과 배치를 제한한다.

이게 사주는 건 프로그래밍 모델이다. 웨이퍼 하나가 한 레이어의 activation 전체를 쥐므로 tensor parallelism도, pipeline parallelism도, FSDP sharding도 없다 — 70B 모델이 단일 디바이스 프로그램으로 쓰이고, 다중 시스템 확장은 SwarmX(weight 스트림을 N개 웨이퍼로 fan-out하고 돌아오는 길에 gradient를 합산하는 broadcast/reduce 트리)를 통한 순수 data parallelism뿐이다. GPU 학습을 지배하는 parallelism 전략 스프레드시트에 Cerebras 페이지는 아예 없는 셈이다.

대가는 규모이고, 시장의 선택이 그걸 드러낸다. 스펙시트는 CS-3 2,048대를 말하지만 공개된 최대 클러스터는 64대(Condor Galaxy 3)다. 플랫폼에서 from-scratch로 학습된 최대 모델은 Jais 2 — 70B 파라미터, 2.6T 토큰 — 로, 앵커 고객 G42가 Cerebras 엔지니어를 상주시켜 돌렸다. CS-1 이후 7년간, 누구도 70B를 넘긴 적이 없다. 그리고 GPU 랩들이 관례적으로 35~45%로 공개하는 MFU는 Cerebras run에 대해 한 번도 공개된 적이 없다.

### Numerics

한 문장으로 끝난다 — FP16과 BF16(FP32 누산), 그리고 WSE-3부터 Hot Chips 공개에서 fixed-point로 라벨한 16-wide 8비트 정수 경로. FP8 없음, FP4 없음, microscaling 없음. 다른 모든 벤더가 세대마다 정밀도를 반으로 자르고 block scaling으로 정확도를 되사는 동안 Cerebras는 여전히 16비트로 계산하고 그걸 품질 차별화("원본 16비트 weight")로 마케팅한다. 긴장은 명백하다 — SRAM 용량이 이 아키텍처의 가장 희소한 자원인데, 8비트 weight면 모델당 필요한 웨이퍼 수가 절반이 된다. 16비트 전용이 수치적 신념인지 datapath 로드맵의 공백인지는 열린 질문이다.

### Scaling

여기서는 scale-up/scale-out의 의미가 다르다. NVIDIA의 scale-up 문제(72개 패키지를 한 디바이스처럼 행동하게 만들기)는 WSE에서 리소그래피가 풀어버린다 — coherent domain이 팹에서 통째로 출하된다. 남는 건 웨이퍼 가장자리 너머 전부이고, 어떤 기계도 자기 가장자리에 이만큼 세게, 이만큼 일찍 부딪히지 않는다.

scale-up은 웨이퍼 그 자체다. 2D mesh 위 900,000 core, 32비트 링크, 단일 사이클 hop, 24개 color로 정적 라우팅, native broadcast, fabric 총대역 214 Pbit/s. 300 mm 웨이퍼 크기에 46,225 mm²로 고정돼 있다. scale-out은 즉시 Ethernet이다 — 시스템당 12×100 GbE(1.2 Tb/s). 학습은 SwarmX(RoCE 위 data-parallel broadcast/reduce), 추론은 레이어 경계에서 모델을 쪼개 pipeline-parallel로 처리한다.

웨이퍼 내부 fabric에는 SerDes도, 케이블도, transceiver도, 링크당 한계비용도 없다 — 라우팅은 컴파일되고, hop은 1 사이클이며, broadcast는 switch 기능이 아니라 native fabric primitive다. NVL72가 구리 5,184가닥과 NVSwitch ASIC 트레이를 써서 72 GPU에 130 TB/s all-to-all을 주는 자리를, WSE는 단일 리소그래피 객체로 대체한다. 함정은 domain 크기가 상수라는 것이다. NVIDIA의 scale-up domain은 세대마다 자라지만(3년에 NVL72→NVL576) 웨이퍼는 2019년부터 46,225 mm²였고 앞으로도 그렇다. 300 mm가 업계 최대 웨이퍼이므로(450 mm 전환은 10년 전에 죽었다) Cerebras의 scale-up 로드맵은 다음 노드가 주는 밀도가 전부다. 더 가져올 면적이 없다.

추론은 weight streaming을 아예 포기한다 — 산수가 치명적이기 때문이다. 70B 모델의 140 GB를 토큰마다 MemoryX에서 약 150 GB/s 파이프로 스트리밍하면 토큰당 약 1초가 든다. 그래서 추론은 weight를 SRAM에 주차하고 레이어 경계에서 웨이퍼에 걸쳐 모델을 쪼갠다 — Llama 70B를 "최소 4대"의 CS-3에 pipeline-parallel(Ethernet 경유)로, 웨이퍼 추가 한 대당 weight+KV 44 GB와 23 kW 부하가 붙는다.

속도는 진짜다. Artificial Analysis 실측으로 2024년 8월 출시 시점 Llama 3.1 8B 1,850 tok/s, 70B 446 tok/s, Llama 405B 969 tok/s(TTFT 240 ms), 2025년 Llama 4 Maverick 2,522 tok/s — 당대 최고 공개 Blackwell 수치의 약 2.4배다. per-user decode 속도에서 근접하는 GPU 사업자가 없다.

경제성은 날카로운 모서리다. 웨이퍼당 44 GB라 frontier급 모델은 fleet을 먹는다 — SemiAnalysis 추정으로 1.6T급 모델에 CS-3 약 24대(GPU 랙 몇 개면 되는 모델), 시스템당 BOM 추정 약 $450k에 리스트 가격은 $2~3M 선(공식 비공개)이다. decode 중에는 웨이퍼의 거대한 FLOPs가 대부분 논다. Cerebras는 배치 크기 공개를 거부했고 시스템당 throughput을 낸 적이 없다. 같은 오픈 모델 기준 토큰당 API 가격이 GPU 기반 사업자의 약 3~5배다. Llama 405B는 조용히 API에서 내려갔고, SemiAnalysis는 이를 서빙 경제성이 안 맞았던 것으로 읽는다. 고정 SRAM이 context에도 값을 매긴다 — KV cache가 weight와 같은 44 GB에 살아서 긴 context가 용량을 훔치고 replica당 시스템 수를 늘린다. API는 131K 토큰에서 막히는데 frontier 사업자들은 256K~1M을 서빙한다. MoE도 서빙하지만(Qwen3-235B 약 1,500 tok/s, 벤더 수치) 이 포맷에는 최악의 경우다 — 거대한 파라미터 발자국을 한 번에 expert 몇 개만 건드리며, 그걸 가장 비싼 메모리에 얹어둬야 한다.

시장은 이걸 정직하게 가격에 반영했다. Mistral Le Chat(약 1,100 tok/s), Perplexity Sonar, Meta Llama API가 레이턴시 값을 치르고, 2026년 1월 OpenAI가 2028년까지 CS-3 용량 750 MW를 계약했다(체결 시 100억 달러 이상 보도, 이후 200억 달러 초과로 증가) — wafer-scale이 받은 최대 규모의 보증이다. 그 용량 위에서 처음 출하된 플래그십이 2026년 7월 GPT-5.6 Sol(공개 수치 750 tok/s)이다.

### Software — kernel matcher

TPU처럼 compiler-driven이지만 훨씬 좁은 구멍이다. Cerebras 컴파일러는 범용 코드 생성기가 아니라 kernel matcher다. `cerebras.pytorch`가 학습 스텝을 lazy tensor로 추적해 Torch-MLIR과 그래프 IR로 넣고, subgraph를 손으로 쓴 kernel 라이브러리와 매칭하며, 매칭 실패 op은 느린 자동 생성 버전으로 떨어진다. 문서화된 제약은 GPU 기준으로 가혹하다 — 정적 그래프만, dynamic shape 불가, 데이터 의존 control flow 불가, 스텝 중간 eager tensor 접근 불가, PyTorch 버전은 upstream보다 뒤에 고정된다.

그리고 kernel 탈출구가 없다. 새로운 attention 변종에 대한 CUDA의 답은 "kernel을 써라", TPU의 답은 Pallas, ROCm의 답은 Triton이다. Cerebras ML 스택에는 사용자 kernel 경로가 아예 없다 — matcher가 크게 빗나가면 해법은 Cerebras 엔지니어다. 별도 SDK 언어 CSL이 raw machine(task, wavelet, color)을 노출하고 인상적인 HPC 결과를 냈지만(TotalEnergies stencil 코드가 A100 대비 약 228배, CS-2 48대로 Gordon Bell 파이널리스트) PyTorch 흐름과 단절된 별세계다. 플랫폼의 모든 플래그십 모델(Jais, BTLM, Med42)이 Cerebras 직원 상주 하에 공동 개발됐다.

여기에는 묘한 면역이 있다. FlashAttention은 memory hierarchy를 가로질러 attention을 tiling하는 기법인데, WSE에는 tiling할 계층이 없다 — AMD에게 수년의 포팅 지연을 물리는 최적화 부류가 여기서는 아예 적용되지 않는다. 그러나 면역과 빈곤은 같은 사실이다. CUDA에서 복리로 쌓이는 서드파티 kernel 생태계가 여기엔 붙을 표면이 없고, 플랫폼 역사상 모든 kernel 개선의 저자는 한 명이다.

그래서 웨이퍼는 batch-1 decode 속도라는, 독립 검증되고 정직하게 얻어낸 진짜 니치에 서 있다. 비용보다 레이턴시를 높게 치는 고객이 값을 치른다.

Cerebras가 거는 베팅을 정리하면 이렇다.

- **웨이퍼를 자르지 말라** — die 경계가 업계가 내는 세금(SerDes, interposer, HBM 스택, 케이블, switch)이다. reticle field 84개를 메탈로 꿰매면 경쟁 시스템에서 가장 대역폭 높은 경계 자체가 존재하지 않는다.
- **SRAM만이 메모리다** — 업계에서 가장 가파른 비율로 용량을 대역폭과 맞바꾼다(44 GB에 on-wafer aggregate 21 PB/s). 불균형을 계층 뒤에 숨기는 대신 기계의 균형을 맞춘다.
- **Dataflow core, matrix unit 없음** — wavelet 도착으로 촉발되는 작은 core 90만 개, matmul은 broadcast·FMAC·mesh reduction으로 조립한다. 0 건너뛰기가 특수 모드가 아니라 공짜다.
- **weight가 움직이고 activation은 남는다** — weight streaming이 모델 크기(MemoryX)와 웨이퍼 메모리(44 GB)를 분리하고, 클러스터 스케일링을 순수 data parallelism으로 접는다.
- **throughput이 아니라 latency를 판다** — 토큰마다 모델 전체를 HBM 기반 어떤 기계보다 빨리 재독한다. 토큰당 비용으로 경쟁하지 말고 그 속도를 프리미엄으로 매긴다.

## 6. AWS Trainium — 의도적인 fast-follower

Nitro 카드와 Graviton CPU를 만든 Annapurna Labs가 Trainium을 fast-follower로 설계했다. 연산 코어는 TPU의 검증된 플레이북(128×128 weight-stationary systolic array, software-managed scratchpad, 전체 프로그램 컴파일)을 Google의 XLA 컴파일러를 그대로 공유하는 수준까지 가져왔다. scale-out fabric은 이미 AWS 나머지를 나르는 Nitro offload 네트워크다. 진짜 Amazon의 것은 좁고 의도적이다 — 빌려온 코어에 볼트로 붙인 전용 collective 통신 실리콘, 그리고 AWS 안에서만 NVIDIA를 이기면 되도록 가격을 매길 수 있는 수직통합이다.

### 계보

| 연도 | 제품 | 핵심 |
| --- | --- | --- |
| 2015 | Annapurna Labs 인수 | 약 3.5억 달러, AWS의 사내 실리콘 팀이 됨 |
| 2018 | Graviton + Nitro | Arm 서버 CPU와 DPU offload fabric |
| 2019 | Inferentia (NeuronCore-v1) | 첫 ML 칩, 추론 전용 |
| 2022 | Trainium1 (v2) | 첫 학습 칩, NeuronCore-v2 2개, GPSIMD 엔진, HBM 32 GB, NeuronLink 2D torus |
| 2024 | Trainium2 (v3) | NeuronCore 8개, 첫 실질 FP8 가속, HBM3 96 GB, 64칩 UltraServer. Project Rainier 구동 |
| 2025 | Trainium3 (v4) | AWS 첫 3 nm(TSMC N3P), OCP MXFP8/MXFP4, torus를 대체하는 NeuronSwitch all-to-all fabric, 144칩 UltraServer |

### 아키텍처 — 같은 논지, 다른 조립

밑에 깔린 베팅은 TPU와 같지만(software-managed SRAM에서 먹이는 systolic array, 컴파일러의 사전 스케줄, cache 없음, thread scheduler 없음) 단위를 조립하는 방식이 다르다. Trainium 칩은 적은 수의 NeuronCore(Trn1 2개, Trn2·Trn3 8개)를 싣고, NeuronCore 하나는 monolithic matmul 엔진이 아니라 느슨하게 결합된 전문 엔진 묶음이다.

- **Tensor Engine** — 128×128 systolic array
- **Vector Engine** — reduction (layernorm, softmax, pooling)
- **Scalar Engine** — pointwise (activation, GELU)
- **GPSIMD Engine** — 512-bit vector processor 8개, C로 프로그래밍. 위 셋 어디에도 안 맞는 것 전부
그 주위에 데이터 무버가 붙는다 — DMA 엔진 128개, 전송을 시퀀싱하는 Sync Engine, 그리고 Trn2부터 collective 전용 CC-Core다. warp도 wavefront도 없고, 엔진들은 정적으로 스케줄된 dataflow 파이프라인으로 돈다. 하중을 받는 설계 결정은 systolic array 자체가 아니라 그 주위에 무엇을 두는가에 있다.

Tensor Engine은 128×128 PE 격자(MAC 16,384개)를 weight-stationary로 돌린다 — 한쪽 operand 타일을 배열에 적재해 고정(`LoadStationary`)하고 다른 쪽을 흘린다(`MultiplyMoving`). partial sum은 PSUM이라는 작은 accumulator SRAM에 떨어지는데, 엔진이 read-add-write할 수 있어서 128보다 긴 contraction도 K축을 따라 접혀 들어간다. NVIDIA는 이 타일 MMA를 warp 계층에 싸고, Google은 VLIW bundle에서 issue하고, Trainium은 named scratchpad에 대한 명시적 명령 두 개로 노출한다.

배열은 3세대 내내 물리적으로 128×128 고정이다. 바뀌는 건 cell당 몇 개의 곱을 채워 넣느냐다. Trn1의 v2는 BF16/FP16(FP32 누산), FP8은 BF16과 같은 속도(가속 없음)였고, Trn2의 v3는 FP8을 double-pump해 실효 256×128을 제시했다(8비트에서 진짜 2배). Trn3의 v4는 microscaling operand를 채워 실효 512×128, BF16 대비 4배를 낸다. 물리적 MAC 수는 한 번도 안 움직이고, datapath가 더 좁은 숫자를 먹일 뿐이다.

나머지 세 엔진이 배열을 바쁘게 유지한다. 잘 컴파일된 스텝은 넷을 전부 겹친다 — Tensor Engine이 matmul을 갈 때 Vector Engine이 이전 타일 softmax를 돌리고 DMA 엔진이 다음 타일을 준비한다. TPU와 GPU의 attention kernel을 효율적으로 만드는 것과 같은 producer/consumer overlap을, 여기서는 warp나 VLIW 슬롯이 아니라 물리적으로 분리된 엔진으로 표현한다. 대가는 가장자리에서 나온다 — 어떤 전문 엔진에도 안 맞는 연산자는 프로그래머블 GPSIMD 경로로 떨어지고 느리다. 새로운 아키텍처에서 병목이 될 가능성이 가장 높은 부분이며, 비-GPU 가속기가 다 지불하는 long-tail 비용의 Trainium 버전이다.

### 메모리 — 세 계층, 전부 software-managed

AWS 자체 문서가 대조를 그린다 — CPU나 GPU와 달리 NeuronCore에는 cache가 없고 "모든 메모리 이동이 프로그램 안에 명시적"이다.

- **HBM**: Trn1 32 GB, Trn2 96 GB HBM3, Trn3 144 GB HBM3e
- **SBUF (State Buffer)**: 메인 scratchpad, HBM 대비 약 20배 대역폭, 128 파티션, NeuronCore당 24 MiB(v2) / 28 MiB(v3) / 32 MiB(v4)
- **PSUM**: 2 MiB, matmul 출력 전용 accumulator

데이터는 `HBM → SBUF → Tensor Engine → PSUM → SBUF`로 움직이고 모든 hop을 컴파일러가 발행한다. 하드웨어의 prefetch도 eviction도 없다. Google의 VMEM 베팅 그대로이며, 천장과 취약성을 함께 물려받는다.

용량으로는 못 이기니 가격으로 싸운다는 게 실제 전략이다. 설계가 peak FLOPs 대비 넉넉한 HBM 예산을 쓰므로 연산 단위당으로는 비슷한 NVIDIA 부품보다 메모리를 많이 진다. 하지만 절대 용량에서는 뒤진다 — Trn2의 96 GB는 H200·B200 아래, Trn3의 144 GB(2025)는 함께 출하되는 B200 192 GB·B300 288 GB 아래다. 그래서 AWS가 큰 모델 서빙 경제성을 논할 때 실제로 당기는 레버는 메모리 리더십이 아니라 가격 — 자기가 만들고 자기가 임대하는 실리콘 위에서의 연산·HBM 단위당 비용이다.

### Numerics

동일한 정밀도 반감 곡선(FP32 → BF16 → FP8 → FP4)에 Trainium 고유의 주름이 둘 있다. 하나는 configurable FP8이다 — Hopper처럼 E4M3/E5M2로 고정하지 않고 exponent bias를 조정 가능하게 해 E5M2, E4M3, E3M4를 지원한다. 컴파일러가 tensor 단위로 range와 precision을 맞바꿀 수 있다. 다른 하나는 Trn3의 FP4가 throughput을 사주지 않는다는 점이다 — OCP MXFP4 operand가 배열에 닿기 전에 MXFP8로 up-convert되므로 FP4는 FP8 속도로 돌고 메모리·대역폭만 아낀다. 연산은 아니다.

두 세대 모두 업계의 정확도 회복 기법에 기댄다 — Trn3부터 microscaling block exponent, 그리고 전 세대 하드웨어 stochastic rounding이다. 믿지 말아야 할 수치도 하나 있다 — AWS는 4× FP8을 헤드라인으로 쓰는데 자사 아키텍처 문서는 dense FP8 대비 2×라고 적는다(4×는 dense BF16 기준). 마케팅 가속비와 datapath가 정확히 일치하지 않는다.

### Collectives in silicon — GPU에 깔끔한 대응물이 없는 블록

분산 학습과 추론은 wall-clock의 상당 부분을 collective에 쓴다 — 모든 gradient step이 all-reduce, 모든 MoE 레이어가 all-to-all이다. GPU에서는 그 collective가 수학을 돌리는 바로 그 SM 위에서 NCCL kernel로 돌기 때문에 통신과 연산이 같은 실리콘을 놓고 경쟁하고 overlap은 소프트웨어로 쟁취해야 한다.

Trainium은 그 기능을 전용 하드웨어로 깎아냈다 — Trn2 칩당 CC-Core 20개가 NeuronLink 포트에 직결되어 all-reduce, all-gather, reduce-scatter, all-to-all을 Tensor·Vector 엔진이 계속 도는 동안 실행한다. Google이 SparseCore로, Cerebras가 off-core zero filter로 한 것과 같은 수다 — 메인 엔진이 틀린 모양인 워크로드를 찾아, 코어에서 사이클을 훔치는 대신 옆에 작은 전용 블록을 두는 것. 통신이 "멈춰서 하는 일"이 아니라 "동시에 하는 일"이 된다.

### Scaling

NeuronLink로 칩들을 묶은 단일 도메인이 scale-up이다. Trn2는 64칩 UltraServer(torus형 NeuronLink)이고, Trn3는 torus를 NeuronSwitch all-to-all fabric으로 대체하고 144칩으로 확대했다. 역할상 NVL72 / TPU superpod의 대응물이다.

scale-out은 EFA + SRD다 — AWS의 기존 네트워크를 그대로 재사용한다. Nitro가 offload하는 EFA에 SRD transport를 쓴다 — packet spraying 기반 RDMA로, AWS의 나머지를 이미 나르고 있는 바로 그 물건이다. InfiniBand는 없다. 대규모 사례가 Anthropic용 Project Rainier다.

여기가 Trainium 전략의 핵심이다 — 새 fabric을 발명하지 않는다. scale-up은 NeuronLink/NeuronSwitch로 칩을 묶고, 그 바깥은 이미 검증된 클라우드 네트워크에 얹는다. 칩이 상품이 아니라 클라우드가 상품이고 칩은 부품이기 때문에 가능한 선택이다.

*(Scaling·Software 두 절은 원문 본문을 도구 제약으로 끝까지 확보하지 못해, AWS 공식 문서와 원문 계보 표에서 확인된 사실로 보강했다. 자세한 사정은 맨 아래 출처 절에 적어뒀다.)*

### Software — Neuron SDK

Neuron Compiler가 XLA(OpenXLA) 위에 서고, 프런트엔드는 `torch-neuronx`(PyTorch)와 JAX다. 즉 TPU와 같은 compiler-driven 모델을 그대로 채택했다 — 컴파일러가 SBUF/PSUM 배치와 DMA 스케줄을 전부 정한다.

NKI(Neuron Kernel Interface)가 탈출구다 — TPU의 Pallas, GPU의 Triton에 해당하는 kernel 작성 경로로, 컴파일러가 새로운 attention 변종이나 fused MoE dispatch의 최적해를 못 만들 때 쓴다. Cerebras와 결정적으로 다른 지점이 여기다 — Trainium에는 사용자 kernel 경로가 존재한다.

NxD(NeuronX Distributed)가 tensor/pipeline parallelism과 collective를, vLLM 통합이 서빙을 담당한다. 약점은 다른 비-GPU 가속기와 같다 — 연산자가 전문 엔진에 안 맞으면 GPSIMD로 떨어지고, CUDA 생태계의 long tail 코드는 손으로 옮겨야 한다.

Trainium이 거는 베팅을 정리하면 이렇다.

- **클라우드가 상품이고 칩은 부품이다** — Annapurna가 칩·서버·랙·Nitro 네트워크·클라우드 API를 하나의 스택으로 설계하므로, Trainium은 merchant silicon 스펙시트가 아니라 AWS 안에서의 price-performance만 이기면 된다.
- **연산 논지는 빌리고 재발명하지 않는다** — 128×128 weight-stationary 배열, SBUF/PSUM scratchpad, 전체 프로그램 컴파일은 TPU의 베팅이며 Google의 OpenXLA까지 공유한다. 아낀 노력은 네트워크와 랙에 쓴다.
- **collective는 실리콘에 속한다** — 전용 CC-Core가 all-reduce·all-to-all을 하드웨어에서 연산과 겹친다. matmul 유닛의 FLOPs를 훔치는 kernel로 돌리는 대신이다.
- **클라우드 자신의 네트워크를 재사용한다** — scale-out은 SRD transport를 쓰는 EFA다. AWS의 나머지를 이미 돌리고 있는 Nitro offload, packet-sprayed RDMA 그대로다.

## 7. Groq LPU

이 섹션은 원문 해당 부분을 확보하지 못해서 Groq의 Hot Chips 공개 자료와 공개 기술 문헌으로 재구성했다. 원문이 확정적으로 밝힌 건 서두의 한 문장뿐이다 — Groq LPU는 200억 달러 규모 acquihire로 NVIDIA에 흡수됐다(별도로 Groq는 NVIDIA와 자사 inference 기술에 대한 비독점 라이선스 계약을 발표했다).

TSP(Tensor Streaming Processor)는 "하드웨어에서 비결정성을 제거하면 컴파일러가 전 사이클을 계획할 수 있다"는 극단적 베팅이다. cache도, speculative execution도, branch predictor도, core 간 통신 중재도 없다. 제어를 전부 컴파일러로 넘긴 software-defined hardware다. 창업자 Jonathan Ross가 Google 원년 TPU 설계자였다는 점을 생각하면 계보가 분명하다 — TPU의 compiler-first 논지를 더 밀어붙여, 네트워크까지 결정론적으로 만든 것이다.

아키텍처의 요점은 이렇다. deterministic execution — 명령 실행 시점이 컴파일 시점에 정적으로 확정된다. 실행 시간을 사전에 정확히 예측할 수 있어 런타임 최적화가 필요 없고, 레이턴시 분산(tail latency)이 사실상 사라진다. 공간적 기능 유닛 + chaining — 기능 유닛들이 공간적으로 배치되어 한 유닛의 출력이 인접 다운스트림 유닛의 입력에 직결된다. 연산 종류는 크게 matrix(MXM), vector(VXM), switch(SXM — transpose·permute 등 데이터 재배치), memory(MEM), 그리고 instruction control이다.

MXM은 320 lane × 320 feature 구조다. plane당 weight 102,400개, MACC 409,600개를 온칩에 보유하고, INT8 기준 약 750 TOPS(900 MHz), mm²당 1 TeraOps 이상의 연산 밀도를 낸다.

메모리는 SRAM only다 — DRAM도 HBM도 쓰지 않는다. 칩당 약 230 MB SRAM, 약 80 TB/s 대역폭에 stride에 둔감하다. Cerebras와 같은 계열의 베팅을 훨씬 작은 die에서 한 셈이다.

네트워킹은 칩당 약 480 GB/s다. 칩·PCIe 카드·서버·네트워크까지 전부 결정론적으로 설계했고, Dragonfly 토폴로지를 써서 비결정성을 만들 수 있는 요소를 제거한다. 즉 네트워크조차 컴파일러가 스케줄한다 — 모든 보드의 모든 칩이 동기화되어 돈다.

칩당 230 MB라는 숫자가 모든 걸 규정한다. 70B 모델 하나를 서빙하려면 수백 개 LPU에 레이어를 쪼개 얹어야 한다. 그래서 Groq의 scale-up/scale-out 구분은 흐릿하다 — 딥 파이프라인 자체가 모델 하나이고, 결정론적 네트워크가 그 파이프라인을 하나의 거대한 정적 스케줄로 묶는다.

보상은 명확하다. HBM 왕복이 없으므로 batch-1 decode에서 GPU를 크게 앞선다 — 출시 초기 Llama-2 70B에서 사용자당 300 tok/s 이상, 이후 세대에서 500 tok/s 이상을 실측으로 보여줬고, 이게 "LLM 응답이 실시간으로 느껴지는" 체감을 만들었다. 대가도 Cerebras와 같은 종류다 — 모델당 필요한 실리콘 수가 많고, 자본 집약적이다. 그래서 Groq는 2024년부터 칩을 제3자에게 파는 것을 그만두고 GroqCloud로 inference-as-a-service에 집중했다(사우디 등과의 데이터센터 규모 파트너십 포함). 2세대 LPU는 Samsung의 Taylor, Texas 팹에서 4 nm(SF4X)로 생산한다.

컴파일러가 전부다. PyTorch/ONNX에서 모델을 받아 GroqCompiler가 사이클 단위 정적 스케줄을 생성한다. 사용자가 손으로 kernel을 튜닝할 여지가 원래 거의 없다 — 이게 장점(결정론, push-button 배포)이자 한계(새로운 연산자에 대한 대응 속도)다. 행렬 크기를 바꿔가며 측정해도 80% 이상 utilization을 꾸준히 유지한다는 게 Groq 측 주장이며, 이는 GPU의 가변 utilization과 대비되는 결정론적 설계의 직접적 결과다.

Groq가 거는 베팅을 정리하면 이렇다.

- **비결정성을 전부 제거한다** — cache, speculation, branch prediction, 동적 중재를 지우면 컴파일러가 전 사이클을 계획할 수 있고 tail latency가 사라진다.
- **DRAM/HBM을 아예 쓰지 않는다** — 칩당 230 MB SRAM, 80 TB/s. memory wall을 우회하는 대신 모델을 칩 수백 개에 펼친다.
- **네트워크까지 결정론적으로** — Dragonfly 토폴로지로 라우터의 비결정성을 제거해, 전체 시스템을 하나의 동기 기계로 만든다.
- **throughput이 아니라 per-user latency를 판다** — Cerebras와 같은 시장 포지션으로, 속도에 프리미엄을 낼 고객을 겨냥한다.
- **칩을 팔지 말고 토큰을 팔라** — 자본 집약적 구조를 인정하고 GroqCloud로 수직통합한다.

## 8. 횡단 비교

### 8-1. 네 가지 질문으로 본 여섯 아키텍처

| | 데이터가 사는 곳 | 연산 유닛 | 스케줄러 | scale-up domain |
| --- | --- | --- | --- | --- |
| NVIDIA | HBM + HW cache(L2/L1) + SMEM·TMEM | Tensor Core (warp 추상화 뒤) | HW warp scheduler + SW pipelining | NVL72 → NVL144 → NVL576 (coherent) |
| Google TPU | HBM + VMEM/CMEM/SMEM, cache 없음 | MXU systolic (128/256²) | XLA 컴파일러 (VLIW) | superpod 9,216 → 9,600 (message-passing) |
| AMD | HBM + 256 MB Infinity Cache + LDS | Matrix Core (wave64 scope) | HW scheduler | 8-GPU OAM → Helios 72 (UALink) |
| Cerebras | SRAM 44 GB 단일 계층 (+ MemoryX) | matrix unit 없음 — dataflow core 90만 | 데이터 도착 = 스케줄 (+ kernel matcher) | 웨이퍼 1장 (46,225 mm², 고정) |
| AWS Trainium | HBM + SBUF/PSUM, cache 없음 | Tensor Engine 128² + Vector/Scalar/GPSIMD | Neuron/XLA 컴파일러 | UltraServer 64 → 144 (NeuronSwitch) |
| Groq | SRAM 230 MB, DRAM 없음 | MXM 320×320 + VXM/SXM | 컴파일러 (네트워크까지) | 결정론적 딥 파이프라인 (Dragonfly) |

### 8-2. 소프트웨어 탈출구의 유무

| 플랫폼 | 기본 경로 | kernel 탈출구 | FA4급 최적화 대응 |
| --- | --- | --- | --- |
| NVIDIA | CUDA kernel (직접) | CUDA/PTX/CUTLASS — 무제한 | 최초 구현이 여기서 나온다 |
| AMD | PyTorch → Triton/CK/HIP | Triton, Composable Kernel, AITER | 수개월 지연 (FA4는 아직 없음) |
| Google TPU | JAX → XLA (전자동) | Pallas (+ Mosaic) | Pallas로 직접 작성 가능 |
| AWS Trainium | PyTorch/JAX → Neuron/XLA | NKI | NKI로 작성, GPSIMD fallback은 느림 |
| Cerebras | cerebras.pytorch → kernel matcher | 없음 (CSL은 별세계) | 계층이 없어 개념상 불필요 / 대신 Cerebras 엔지니어 의존 |
| Groq | ONNX/PyTorch → GroqCompiler | 사실상 없음 | 컴파일러 개선 대기 |

### 8-3. 네트워킹 스택 비교

| 플랫폼 | scale-up fabric | coherent? | scale-out | 광 전략 |
| --- | --- | --- | --- | --- |
| NVIDIA | NVLink + NVSwitch crossbar | 예 (cache-coherent) | InfiniBand(Quantum-X) / Spectrum-X Ethernet + ConnectX·BlueField | 랙 내 passive copper → 랙 간 광 → Rubin에서 CPO |
| Google TPU | ICI 2D/3D torus | 아니오 (message-passing) | Virgo(east-west) + Jupiter(north-south) | OCS(Palomar/Apollo) — 랙부터 건물 spine까지 동일 primitive |
| AMD | Infinity Fabric → UALink(출시 시 UALoE) | 예(목표) / 출시 시 터널링 | Ethernet + UEC/UET, Pensando NIC | Broadcom Tomahawk 6 CPO — 파트너 실리콘 |
| Cerebras | on-wafer 2D mesh (SerDes 없음) | 해당 없음 (단일 칩) | 12×100 GbE, SwarmX / RoCE | - |
| AWS Trainium | NeuronLink torus → NeuronSwitch | 아니오 | EFA + SRD (Nitro offload, Ethernet) | - |
| Groq | 결정론적 Dragonfly | 아니오 | 동일 fabric (구분 흐림) | - |

## 9. 여섯 아키텍처를 관통하는 7가지 패턴

여섯 회사를 따로 읽고 나면 겹치는 패턴이 보인다. 정리하면 이렇다.

**① 정밀도 반감 + 더 촘촘한 scaling.** FP32 → FP16 → FP8 → FP4로 세대마다 비트를 반으로 자르고, block 단위 shared exponent(OCP MX, NVFP4)로 정확도를 되산다. NVIDIA MXFP4, AMD OCP MX, TPU v8 MXU에 들어가는 포맷이 동일한데, OCP가 AMD·NVIDIA·Intel·Meta·Microsoft·Qualcomm·ARM이 참여한 개방 컨소시엄에서 규정했기 때문이다. 여기에 hardware stochastic rounding이 거의 모든 플랫폼에 들어갔다. 예외는 Cerebras(16비트 고수)와, FP4가 연산 이득을 주지 않는 Trainium3다.

**② matmul 명령의 비동기화 — 누가 issue하는가.** NVIDIA는 warp 32 → warp-group 128 → 단일 스레드 + descriptor로 걸어왔다. 흥미로운 건 종착점이 이미 존재했다는 것이다 — Cerebras core에서 tensor 명령은 애초에 DSR descriptor 형태밖에 없었다. NVIDIA는 다섯 세대에 걸쳐 그 지점으로 걸어갔고, AMD는 wave64 scope에 머물러 이 궤적을 따라오지 않았다. 그 차이가 attention kernel(matmul과 softmax의 overlap)에서 직접 비용으로 나타난다.

**③ compiler-driven vs kernel-driven, 그리고 수렴.** TPU·Trainium·Cerebras·Groq는 컴파일러가 스케줄러다(cache 없음, 동적 스케줄러 없음, 틀리면 fallback 없음). NVIDIA·AMD는 kernel-driven이다. 하지만 Triton과 `torch.compile`이 GPU 쪽에서 격차를 좁히고, Pallas·NKI가 가속기 쪽에서 kernel 경로를 열면서 양쪽이 수렴 중이다. 그래도 철학적 양극은 남는다 — TPU에서 컴파일러는 유일한 인터페이스고, GPU에서는 여럿 중 하나다.

**④ "메인 엔진이 틀린 모양인 워크로드는 옆에 전용 블록을 깎아라".** Google SparseCore(embedding lookup) → CAE(decode 중 collective reduction), AWS CC-Core(all-reduce/all-to-all), Cerebras 송신 측 zero filter. 셋 다 같은 수다 — 메인 코어를 비틀어 맞추지 말고, 작은 면적을 떼어 전용 블록을 옆에 둔다. GPU에서 collective가 수학을 돌리는 바로 그 SM 위에서 NCCL kernel로 돌며 실리콘을 놓고 경쟁하는 것과 정확히 대비된다.

**⑤ scale-up domain 확대 경쟁 — 그리고 Cerebras만 자라지 않는다.** NVIDIA 72 → 144 → 576 GPU(3년), TPU superpod 4,096 → 9,216 → 9,600칩, AMD 8-GPU 박스 → Helios 72, AWS UltraServer 64 → 144. Cerebras만 2019년 이후 46,225 mm²에 고정되어 있고, 300 mm가 업계 최대 웨이퍼라 더 가져올 면적이 없다. scale-up domain 크기는 곧 "tensor parallelism과 MoE all-to-all을 느린 fabric에 내보내지 않고 가둘 수 있는 한계"다.

**⑥ 배선의 물리학 — copper의 벽, 그리고 CPO.** 200G/lane에서 passive copper DAC는 약 1.5~2 m가 한계다. 그래서 랙 내부는 copper, 랙을 넘으면 광이다. NVIDIA가 NVL576을 위해 Kyber라는 새 섀시를 설계한 이유가 이것이다 — 토폴로지가 아니라 케이블 길이가 랙 형상을 결정했다. 랙 간 구간에서는 pluggable transceiver 수만 개의 laser 전력이 수십 kW라 Rubin 세대에서 Quantum-X/Spectrum-X Photonics가 co-packaged optics로 접어 넣는다(laser 약 1/4, 링크 전력 약 1/3.5). Broadcom Tomahawk 6의 Davisson도 같은 방향이다. Google만 완전히 다른 축에 있다 — 패킷 스위칭이 아니라 ms급 재구성의 circuit switching(OCS)을 랙부터 건물 spine까지 같은 primitive로 쓴다.

**⑦ Ethernet의 부상 — UET은 RoCEv2가 아니다.** Dell'Oro 기준 2025년 AI scale-out fabric 물량에서 Ethernet이 InfiniBand의 2배 이상이다. AWS, Microsoft, Meta, Oracle, xAI 모두 Ethernet으로 표준화했고, AMD는 InfiniBand를 아예 팔지 않는다. 핵심은 UEC 1.0의 UET이다 — packet spraying, SACK 기반 selective retransmission, 현대적 congestion control을 갖는다. RoCEv2가 InfiniBand transport를 Ethernet 프레임에 캡슐화한 것이라면, UET은 RDMA 시맨틱을 scale-out AI fabric용으로 다시 설계한 것이다. NVIDIA도 Spectrum-X로 Ethernet 선택지를 제공한다. 남은 질문은 "Ethernet이 IB를 따라잡는가"가 아니라 "rack-scale coherent domain 격차를 누가 먼저 닫는가"로 옮겨갔다.

## 10. 네트워크 엔지니어 관점에서 건질 것

이 섹션은 원문에 없는, 내가 네트워크 엔지니어로서 덧붙인 해석이다.

**10-1. "scale-up이냐 scale-out이냐"는 곧 parallelism 배치 문제다.** 여섯 플랫폼 전부 같은 규칙을 따른다 — tensor parallelism과 MoE expert routing(all-to-all)은 scale-up 안에, data/pipeline parallelism은 scale-out으로. 이유는 단순하다. 칩당 scale-up 대역이 scale-out보다 한두 자릿수 크다. Blackwell 기준 NVLink vs ConnectX-8이 한 자릿수, TPU 기준 ICI vs DCN이 두 자릿수다. 즉 "이 워크로드가 내 fabric을 얼마나 때릴 것인가"는 모델 구조가 아니라 parallelism 배치가 결정하고, 그 배치는 scale-up domain 크기가 결정한다.

**10-2. MoE가 토폴로지 선택을 바꾸고 있다.** TPU 8i가 1,024칩 규모에서 3D torus를 버리고 Boardfly(high-radix 계층)로 간 이유가 교과서적이다. torus는 nearest-neighbour collective(ring all-reduce)에 최적이고, MoE는 정반대인 all-to-all이다. 왕복 레이턴시가 최장 hop 쌍에 묶이므로 diameter 16 hop → 7 hop 압축이 곧 성능이다. 데이터센터 fabric 설계에서 "bisection bandwidth"만 보던 관성에서 "diameter와 all-to-all 완료 시간"으로 축이 옮겨가고 있다는 신호로 읽힌다. TPU 8t의 Virgo가 flat two-layer non-blocking(최대 2 switch hop)으로 간 것도 같은 방향이다.

**10-3. straggler 관리가 fabric 기능이 됐다.** Virgo의 multi-planar fault isolation과 sub-millisecond telemetry는 "스케줄러가 straggler를 스텝 망치기 전에 죽이게 한다"는 목적이 명시돼 있다. 동기 SPMD job에서는 가장 느린 노드가 전체 스텝 시간을 정한다. 즉 AI fabric의 telemetry 요구사항은 전통적 DC의 그것과 시간 스케일이 다르다 — 분 단위 모니터링이 아니라 sub-ms 수준이다. 그리고 Google은 OCS로 장애 복구를 토폴로지 재구성으로 처리한다(죽은 cube를 예비 cube로 광학 교체, ICI domain 유지).

**10-4. 표준 정치가 기술 선택을 앞선다.** UALink와 UEC는 둘 다 "구현이 아니라 표준을 소유한다"는 같은 수다. 그런데 UALink는 native switching 실리콘이 2027년까지 없어서 Helios가 출시 시점에 UALoE(IF를 Ethernet에 터널링)로 간다. 표준이 있어도 실리콘이 없으면 제품 일정이 표준을 기다려주지 않는다는 사실이 그대로 드러난 사례다. 반대로 NVIDIA는 NVLink Fusion으로 자기 fabric을 열어 third-party CPU/XPU를 받아들이기 시작했다 — 개방 진영이 표준으로 포위하려 하자, 폐쇄 진영이 선택적으로 문을 연 구도다.

**10-5. 소프트웨어 해자는 네트워크에도 있다.** 원문이 반복해서 짚는 "NVIDIA는 frontier lab 안에 엔지니어를 보낸다"는 논점은 컴퓨팅에만 해당하지 않는다. NCCL은 단순한 라이브러리가 아니라 토폴로지 인식 collective 구현체이고, 새 fabric(UET, NeuronLink, ICI)이 성능을 내려면 collective 라이브러리가 그 토폴로지를 이해해야 한다. fabric 경쟁은 링크 속도가 아니라 collective 라이브러리 성숙도 경쟁이기도 하다.

## 11. 출처와 작성 노트

원문은 Jacob Peake의 ["AI Chip Architectures"](https://www.jepeake.com/ai-chip-architectures)(미러: jacobpeake.com)다. NVIDIA / Google TPU / AMD / Cerebras 섹션과 AWS Trainium의 아키텍처·연산·메모리·numerics·collectives·bets 부분은 원문 본문을 직접 확보해 정리했다.

다만 본문 추출이 도구 쪽 길이 제한으로 Trainium의 마지막 베팅 항목 중간에서 끊겼다. markdown/text 두 가지 추출 방식과 미러 도메인까지 시도했지만 같은 지점에서 잘렸다. 그래서 두 부분은 공개자료로 재구성했다.

- **Trainium의 Scaling / Software** — AWS 공식 문서 기반(UltraServer 64→144칩, NeuronSwitch, EFA+SRD, Neuron SDK, NKI, NxD). 다만 원문 계보 표에서 확보한 "NeuronSwitch all-to-all fabric이 torus를 대체, 144칩 UltraServer"는 원문 근거다.
- **Groq LPU 섹션 전체** — Groq Hot Chips 34 발표 자료(MXM 320×320, SRAM 230 MB / 80 TB/s, 네트워킹 480 GB/s, INT8 약 750 TOPS, mm²당 1 TeraOps 이상)와 공개 기술 문헌 기반이다. 원문이 확정적으로 밝힌 건 서두의 "Groq LPU는 200억 달러 acquihire로 NVIDIA에 흡수됐다"뿐이다.

원문에 결론 섹션이 더 있었을 가능성이 있지만 확인하지 못했다. 정확성이 중요한 용도라면 위 두 섹션은 원문을 직접 읽고 교차 확인하는 걸 권한다. 10장의 네트워크 엔지니어 관점은 원문에 없는 내 해석이고, 원문의 사실관계 위에 세웠지만 결론은 원저자의 것이 아니다.

LLMSO Week 5 Final Assignment로 정리했다.
