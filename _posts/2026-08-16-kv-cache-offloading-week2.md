---
layout: post
title: "append 한 줄이 페타바이트가 되기까지: KV Cache는 왜 결국 스토리지로 내려갔나"
date: 2026-08-16 10:00:00 +0900
categories: [study, llm-serving]
tags: [KV-Cache, LLM-Serving, GPU, Storage, NVIDIA, study-log]
---

## 들어가며

### 먼저 질문 하나

**KV Cache는 학습에서 많이 쓰일까, 추론에서 많이 쓰일까?**

답부터 말하면 **거의 전적으로 추론 쪽**이다. 캐시가 쓸모 있으려면 두 가지 조건이 필요하다.

**조건 1: 토큰을 하나씩 만들어야 한다.** 학습할 때는 정답 문장을 이미 갖고 있다. "안녕하세요"를 학습시킨다면 다섯 글자를 전부 알고 시작하니, 다섯 위치를 **한 번에 병렬로** 계산한다. 순서대로 하나씩 만들어낼 이유가 없다. 반복이 없으니 캐시할 것도 없다.

**조건 2: 가중치가 고정돼 있어야 한다.** 학습은 매 스텝마다 가중치를 갱신한다. 그런데 Key와 Value는 그 가중치로 계산한 값이다. 가중치가 바뀌면 앞서 계산해둔 Key/Value는 그 즉시 틀린 값이 된다. 저장해봐야 못 쓴다.

추론은 정반대다. 가중치는 고정돼 있고, 다음에 무슨 토큰이 나올지 모르니 하나씩 만들어야 한다. 두 조건이 모두 맞아떨어진다.

그래서 이렇게 정리할 수 있다. **KV Cache는 "가중치가 고정된 상태에서 토큰을 하나씩 만들 때"만 의미가 있다.** 학습의 본체인 순전파·역전파 과정에는 등장하지 않는다.

예외가 하나 있긴 하다. RLHF 같은 사후 학습은 중간에 모델이 직접 문장을 만들어보는 단계(rollout)가 들어간다. 그 부분은 형태는 학습이지만 내용은 추론이라 KV Cache를 쓴다. 다만 이것도 결국 "추론이라서 쓴다"는 이야기라 위의 원칙에서 벗어나지 않는다.

이 질문으로 글을 시작하는 이유가 있다. 앞으로 나올 모든 이야기가 여기서 파생되기 때문이다. 캐시가 추론 전용이라는 건, 사용자 요청이 들어올 때마다 캐시가 생기고 요청이 끝나면 갈 곳을 잃는다는 뜻이다. 그리고 사용자는 학습 작업과 달리 **끝없이 들어온다.**

### 그런데 이게 왜 스토리지 문제가 됐을까

[1주차 글](https://youwins-lab.github.io/Blog/study/llm-serving/2026/08/08/microgpt-week1.html)에서 microGPT 코드를 읽다가 어텐션 블록 안에서 이 두 줄을 발견했었다.

```
keys[li].append(k)    # 이전 위치들의 Key를 계속 쌓아둔다
values[li].append(v)  # 이전 위치들의 Value를 계속 쌓아둔다
```

그때는 "아 이게 KV Cache구나" 하고 넘어갔다. 파이썬 리스트에 값을 덧붙이는 평범한 코드였고, 파라미터가 4,192개짜리 장난감 모델이었으니 메모리를 걱정할 일도 없었다. 앞의 질문에 비춰 보면 이 코드가 학습을 돌리는 부분이 아니라 모델이 실제로 글자를 만들어내는 부분에 있었다는 것도 자연스럽다.

2주차 스터디에서 서빙 시스템 설계를 읽으면서 이 두 줄이 다시 나타났다. 이번엔 반가움보다 당황스러움에 가까웠다. 여러 모델을 한 GPU에 올릴 때의 트레이드오프를 이야기할 때도, TTFT나 TPOT 같은 성능 지표를 정의할 때도, agentic 워크로드의 비용 구조를 설명할 때도, 이야기의 중심에 KV Cache가 있었다. 최근에 지인을 통해 들었는데 WEKA, VAST Data, Netapp, DDN 같은 스토리지 회사들이 "KV Cache를 우리 스토리지에 저장해드립니다"라고 하면서 제품을 팔고 있었다.

장난감 코드에서 `append` 한 줄이던 게, 어쩌다 스토리지 벤더들이 전용 제품을 만들 만큼 큰 문제가 됐을까. 이번 주 과제는 그 간극을 메우는 것으로 잡았다.


---

## 1. 왜 KV Cache가 필요한가

### 캐시가 없으면 매번 처음부터 다시 읽는다

1주차에서 봤듯이 어텐션은 "지금 이 토큰을 처리할 때 앞에 나온 모든 토큰을 얼마나 참고할지"를 계산한다. 그러려면 앞선 모든 토큰의 Key와 Value가 필요하다.

문제는 GPT 계열 모델이 토큰을 **하나씩** 만든다는 것이다. 이걸 autoregressive 생성이라고 부른다. "안녕하세요"를 만들려면 '안' → '녕' → '하' → ... 순서로 다섯 번의 독립적인 계산이 필요하다.

여기서 캐시가 없다고 해보자. 5번째 토큰을 만들 때 1~4번째 토큰의 Key/Value를 다시 계산해야 한다. 그런데 1~4번째 토큰은 이미 앞 단계에서 계산했던 것이고 값도 똑같다. 입력이 안 바뀌었으니 결과도 안 바뀐다. 매번 같은 답이 나올 계산을 반복하는 셈이다.

연산량 차이를 정리하면 이렇다. n개 토큰을 생성할 때:

| 구분 | 매 스텝 계산량 | n개 토큰 전체 |
|---|---|---|
| 캐시 없음 | 지금까지의 전체 시퀀스를 다시 처리 | 대략 n의 세제곱에 비례 |
| 캐시 있음 | 새 토큰 1개분만 처리 + 캐시 읽기 | 대략 n의 제곱에 비례 |

"세제곱"이 나오는 이유는, 매 스텝마다 이미 제곱에 비례하는 어텐션 계산을 처음부터 다시 하기 때문이다. 제곱짜리 작업을 n번 반복하니 세제곱이 된다. 컨텍스트가 길어질수록 이 차이는 감당할 수 없어진다. 그래서 KV Cache는 "있으면 좋은 최적화"가 아니라 **없으면 서비스가 성립하지 않는 필수 요소**다.

### Prefill과 Decode는 완전히 다른 작업이다

캐시를 쓰기 시작하면 추론이 성격이 전혀 다른 두 단계로 나뉜다. 이 구분이 이 글 전체의 기준선이다.

**Prefill**은 사용자가 입력한 프롬프트 전체를 한 번에 처리하는 단계다. 프롬프트가 2,000토큰이면 2,000개를 동시에 계산해서 KV Cache를 채운다. 토큰들이 병렬로 처리되니 GPU 연산 유닛이 바쁘게 돌아간다. 이런 상태를 **compute-bound**라고 한다. 계산할 게 많아서 느린 상태다.

**Decode**는 그 뒤로 토큰을 하나씩 만들어내는 단계다. 새로 계산할 건 토큰 하나분뿐인데, 그 하나를 계산하려고 모델 가중치 전체와 KV Cache 전체를 메모리에서 읽어와야 한다. 계산할 건 적고 읽어올 건 많다. 이런 상태를 **memory-bandwidth-bound**라고 한다. GPU 연산 유닛은 놀고 있고 데이터가 도착하기를 기다리는 상태다.

이걸 좀 더 정확하게 표현하는 지표가 **arithmetic intensity**다. "메모리에서 1바이트를 읽어올 때마다 연산을 몇 번 하는가"를 나타낸다. Prefill은 이 값이 높고 Decode는 낮다. 한 논문은 GPU에 이상적으로 맞는 지점을 바이트당 약 295 FLOP 부근으로 잡는데, MLA를 decode 경로로 실행하면 바이트당 약 37 FLOP 수준까지 떨어져 성격이 달라진다고 지적한다([arXiv:2605.15250](https://arxiv.org/abs/2605.15250)). 숫자 자체보다 **같은 모델도 어느 단계냐에 따라 병목이 완전히 달라진다**는 점이 핵심이다.

### 그래서 성능 지표가 여러 개로 갈린다

LLM 서빙 지표가 왜 하나가 아닌지도 이 구분에서 나온다.

- **TTFT** (Time To First Token): 주로 prefill 시간에 좌우된다. 사용자가 "응답이 시작됐다"고 느끼는 순간까지의 시간.
- **TPOT** (Time Per Output Token) 또는 **ITL** (Inter-Token Latency): decode 속도. 응답이 흘러나오는 체감 속도.
- **Throughput**: 전체 시스템이 초당 몇 토큰을 만드는가. batch size에 크게 좌우된다.
- **Goodput**: throughput 중에서 SLO(응답 시간 목표)를 지킨 것만 센 값. 100개를 처리했는데 30개가 너무 느렸다면 goodput은 70이다.
- **Prefix cache hit rate**: 요청이 들어왔을 때 앞부분 KV Cache를 재사용할 수 있었던 비율. 이게 높으면 TTFT가 극적으로 줄어든다. 뒤에서 다룰 offloading이 결국 이 값을 올리기 위한 것이다.

TTFT를 줄이려면 batch를 작게 하는 게 유리하고, throughput을 올리려면 batch를 크게 하는 게 유리하다. 서로 반대 방향이다. 이 줄다리기의 한가운데에 KV Cache가 있다. batch를 키우려면 KV Cache를 그만큼 더 담아야 하기 때문이다.

**여기까지 정리하면**: KV Cache는 중복 계산을 없애기 위해 반드시 필요하고, 그 결과 추론이 prefill과 decode라는 성격이 다른 두 단계로 나뉘며, 성능 지표들도 그 구분을 따라 갈라진다.

---

## 2. KV Cache가 얼마나 큰가 — 직접 계산해보기

1주차에서 microGPT의 파라미터 4,192개를 직접 더해봤던 것처럼 이번에도 숫자를 손으로 계산해보자. 이 절이 이 글에서 가장 중요하다고 생각한다. 뒤에 나오는 모든 이야기가 여기서 나온 숫자 때문에 존재하기 때문이다.

### 계산 공식 만들기

토큰 하나가 차지하는 KV Cache 크기는 이렇게 나온다.

```
토큰당 바이트 = layer 수 × 2 × KV head 수 × head dim × 데이터 타입 바이트
```

각 부분이 왜 필요한지 하나씩 보자.

- **layer 수**: Transformer는 같은 블록을 여러 층 쌓는다. 층마다 자기 KV Cache를 따로 갖는다. 80층이면 80벌이다.
- **× 2**: Key와 Value 두 종류를 저장하니까 2배.
- **KV head 수 × head dim**: 한 층에서 Key 벡터 하나의 크기.
- **데이터 타입 바이트**: FP16이면 2바이트, FP8이면 1바이트.

여기에 시퀀스 길이와 batch size를 곱하면 총량이 된다.

```
총 KV Cache = 토큰당 바이트 × 시퀀스 길이 × batch size
```

이 공식이 맞는지 확인할 수 있는 좋은 예가 있다. VAST Data가 자사 블로그에서 Llama 3.1-405B를 예로 들어 계산한 것을 보면, 126개 layer, KV head 8개, head dim 128, FP16 기준으로 토큰당 크기를 126 × 2 × 8 × 128 × 2 = 516,096바이트로 산출한다. 위 공식과 곱하는 순서만 다를 뿐 완전히 같다.

### Llama 3 70B로 계산해보기

이제 좀 더 흔한 모델로 계산해보자. Llama 3 70B는 layer 80개, KV head 8개, head dim 128이다.

**토큰 1개당 (FP16, 2바이트):**
```
80 × 2 × 8 × 128 × 2 = 327,680 바이트 = 320 KB
```

계산기를 두드려보면 토큰 하나에 320KB다. 이 숫자가 실감이 안 날 수 있는데, 텍스트 한 글자를 처리하는 데 320KB짜리 이미지 한 장 분량의 메모리가 필요하다는 뜻이다.

**컨텍스트 길이별로 (요청 1개 기준):**

| 컨텍스트 길이 | 계산 | KV Cache 크기 |
|---|---|---|
| 4K 토큰 | 320KB × 4,096 | 약 1.3 GB |
| 32K 토큰 | 320KB × 32,768 | 약 10.7 GB |
| 128K 토큰 | 320KB × 131,072 | 약 42 GB |

참고로 같은 모델을 int8(1바이트)로 계산한 값이 [JAX Scaling Book](https://jax-ml.github.io/scaling-book/applied-inference/)에 나와 있는데, 토큰당 160KB, 시퀀스 길이 32K에서 요청 하나당 약 5.3GB로 잡는다. FP16인 내 계산의 정확히 절반이니 서로 맞는다.

### GPU에 몇 명이나 들어갈까

이제 실제 GPU와 비교해보자. H100은 HBM3 80GB에 대역폭 3.35TB/s, H200은 HBM3e 141GB에 4.8TB/s다.

H200 한 장에 Llama 3 70B를 FP8로 올린다고 하면 가중치가 약 70GB를 먹는다. 남는 건 71GB 정도다.

- 4K 컨텍스트: 71GB ÷ 1.3GB ≈ **약 53개 요청**
- 32K 컨텍스트: 71GB ÷ 10.7GB ≈ **약 6~7개 요청**
- 128K 컨텍스트: 71GB ÷ 42GB ≈ **약 1.7개 요청**

(내가 위 공식으로 계산한 값이다. 실제로는 activation과 프레임워크 오버헤드가 더 붙어서 이보다 적다.)

128K 컨텍스트에서는 사실상 동시 요청 **한 개**가 한계다. 수억 원짜리 GPU가 사용자 한 명을 상대한다. 이 숫자를 처음 계산했을 때 좀 놀랐다. 그동안 "GPU가 부족하다"는 이야기를 들으면 막연히 연산 능력 이야기인 줄 알았는데, 긴 컨텍스트 구간에서는 **메모리 용량이 먼저 벽에 부딪힌다**.

그리고 이게 여러 모델을 한 GPU에 올릴 때의 트레이드오프의 실체이기도 하다. 가중치가 자리를 차지하는 만큼 KV Cache에 쓸 공간이 줄고, 그만큼 동시 처리 가능한 요청 수가 줄어든다. "모델을 몇 개 올릴까"는 사실 "동시 사용자를 몇 명 받을까"와 같은 질문이다.

**여기까지 정리하면**: Llama 3 70B 기준 토큰당 320KB, 128K 컨텍스트면 요청 하나에 42GB다. 긴 컨텍스트에서는 모델 가중치가 아니라 KV Cache가 병목이 된다.

---

## 3. GPU 안에서의 절감 기법

42GB라는 숫자를 봤으니 사람들이 이걸 줄이려고 뭘 했는지 보자. 크게 두 갈래다. 하나는 모델 구조 자체를 바꾸는 것(학습 단계에서 결정), 다른 하나는 서빙 소프트웨어에서 메모리를 잘 쓰는 것(배포 단계에서 결정)이다.

### 3-1. 구조를 바꾼다: MHA → MQA → GQA → MLA

앞의 공식에서 줄일 수 있는 건 사실상 "KV head 수"밖에 없다. layer 수와 head dim은 모델 성능에 직결되기 때문이다.

**MHA (Multi-Head Attention)**: 원조 방식. Query head가 64개면 Key/Value head도 64개다. 캐시가 가장 크다.

**MQA (Multi-Query Attention)**: 반대쪽 극단. Query head가 몇 개든 Key/Value는 **딱 한 세트만** 쓴다. 모든 Query head가 같은 Key/Value를 공유한다. Query head 수만큼 캐시가 줄어들지만 대체로 품질 저하가 뚜렷하다. 2019년 Noam Shazeer의 논문에서 제안됐다([arXiv:1911.02150](https://arxiv.org/abs/1911.02150)).

**GQA (Grouped-Query Attention)**: 둘 사이의 절충. Query head를 몇 개씩 묶어서 그룹을 만들고, 그룹 하나가 Key/Value 한 세트를 공유한다. 64개 Query head를 8개 그룹으로 묶으면 캐시가 8분의 1이 된다.

비유하자면 MHA는 팀원 64명이 각자 자기 자료를 들고 다니는 것, MQA는 64명이 공용 자료 하나를 돌려 보는 것, GQA는 8명씩 팀을 짜서 팀마다 자료 한 벌을 쓰는 것이다.

GQA 논문에서 인상적이었던 건 **uptraining**이라는 개념이다. 이미 MHA로 학습이 끝난 모델을 원래 사전학습 연산의 5% 정도만 써서 MQA나 GQA 모델로 변환하는 방법을 제안한다. 이렇게 변환한 GQA가 MQA에 가까운 속도로 MHA에 근접한 품질을 낸다([arXiv:2305.13245](https://arxiv.org/abs/2305.13245)). 모델을 처음부터 다시 학습하지 않아도 된다는 게 실무적으로 큰 차이다.

**MLA (Multi-head Latent Attention)**: DeepSeek-V2에서 나온 방식. head 수를 줄이는 대신 Key/Value를 저차원 latent 벡터로 압축해서 저장한다. DeepSeek-V2 논문은 MQA/GQA가 캐시는 줄이지만 MHA만큼의 성능은 못 낸다고 지적하면서, low-rank KV 압축을 쓴 MLA가 MHA보다 나은 성능을 내면서도 캐시는 더 작다고 주장한다([arXiv:2405.04434](https://arxiv.org/abs/2405.04434)).

정리하면 이렇다.

| 방식 | KV Cache 크기 | 품질 | 대표 모델 |
|---|---|---|---|
| MHA | 기준 (가장 큼) | 기준 | 초기 GPT 계열 |
| MQA | Query head 수만큼 감소 | 저하 있음 | 일부 초기 모델 |
| GQA | 그룹 수만큼 감소 (보통 8배) | MHA에 근접 | Llama 3, Qwen, Mistral 등 |
| MLA | low-rank 압축, 가장 작음 | 논문상 MHA 이상 | DeepSeek-V2/V3 |

한 가지 주의할 점. 이 선택은 **학습할 때 이미 결정**된다. 운영자가 배포 시점에 바꿀 수 있는 게 아니다. 우리가 할 수 있는 건 "어떤 구조의 모델을 고를 것인가"까지다. 위 표의 절감 배수는 그래서 모델 선택 기준으로 읽어야 한다.

### 3-2. 메모리를 잘 쓴다: PagedAttention

여기서부터는 배포 시점에 선택할 수 있는 영역이다.

문제는 이렇다. 요청이 들어올 때 이 사용자가 몇 토큰까지 쓸지 미리 알 수 없다. 그래서 초기 서빙 시스템들은 안전하게 최대 길이만큼 메모리를 미리 잡아뒀다. 최대 2,048토큰을 허용하는 시스템에서 A는 50토큰, B는 300토큰, C는 700토큰을 쓰더라도 셋 다 2,048토큰 분량을 예약받는 식이다.

이렇게 잡아두고 안 쓰는 공간이 생기는 걸 **fragmentation**이라고 한다. 두 종류가 있다.

- **internal fragmentation**: 2,048을 예약했는데 50만 써서 나머지 1,998칸이 그 요청에 묶인 채 놀고 있는 상태.
- **external fragmentation**: 전체적으로는 메모리가 남는데 조각조각 흩어져 있어서 큰 덩어리가 필요한 새 요청을 못 받는 상태.

디스크 조각모음을 떠올리면 된다. 용량은 남는데 연속된 공간이 없어서 큰 파일을 못 쓰는 상황과 같다.

**PagedAttention**은 여기에 운영체제의 virtual memory 개념을 그대로 가져왔다. KV Cache를 고정 크기 블록(page)으로 쪼개고, 물리 메모리에 **연속되지 않게** 흩어서 저장한다. 논리 블록과 물리 블록의 매핑은 block table이 관리한다. 논리 KV 블록과 물리 KV 블록을 분리함으로써, 미리 예약하지 않고도 필요할 때마다 동적으로 캐시를 늘릴 수 있다.

성능 효과는 논문에 이렇게 나와 있다. vLLM은 KV Cache 메모리 낭비를 거의 0에 가깝게 만들고 요청 내·요청 간 캐시 공유를 가능하게 해서, FasterTransformer나 Orca 같은 기존 시스템 대비 같은 지연시간에서 throughput을 2~4배로 끌어올렸다([arXiv:2309.06180](https://arxiv.org/abs/2309.06180), SOSP 2023).

지금은 PagedAttention이 사실상 업계 표준이 되어 TGI, vLLM, TensorRT-LLM에서 모두 지원한다.

다만 공짜는 아니다. 한 서베이 논문은 비연속 메모리 블록에 맞춰 어텐션 커널을 다시 작성해야 하고, 메모리 관리자 자체가 소프트웨어 복잡도와 성능 오버헤드를 더한다는 점을 약점으로 지적한다. 그래서 KV Cache를 연속된 virtual memory에 유지하는 vAttention 같은 대안도 제안됐다.

### 3-3. 같은 앞부분은 한 번만: Prefix Caching과 RadixAttention

또 하나의 큰 절약 지점이 있다. 실제 서비스에서는 여러 요청이 **앞부분을 공유**하는 경우가 아주 많다.

- 같은 system prompt를 쓰는 모든 요청
- multi-turn 대화의 2번째, 3번째 턴 (앞선 대화 내용이 전부 앞부분에 그대로 들어간다)
- 같은 문서를 놓고 여러 질문을 하는 RAG

앞부분이 같으면 KV Cache도 똑같다. 그러면 한 번만 계산해서 공유하면 된다. 이걸 **prefix caching**이라고 한다.

**RadixAttention**(SGLang)은 이 공유 관계를 radix tree로 관리한다([arXiv:2312.07104](https://arxiv.org/abs/2312.07104)). 요청들이 공유하는 앞부분은 트리의 몸통이 되고, 갈라지는 부분부터 가지가 된다. 새 요청이 오면 트리를 따라가면서 어디까지 재사용할 수 있는지 찾는다.

### 3-4. 정밀도를 낮춘다: KV Cache Quantization

가장 단순한 방법이기도 하다. 공식의 맨 마지막에 곱해지는 "데이터 타입 바이트"를 줄이는 것이다. FP16(2바이트)을 FP8(1바이트)로 바꾸면 캐시가 절반, INT4로 가면 4분의 1이 된다.

문제는 품질이다. Key와 Value는 어텐션 계산에 직접 들어가는 값이라 정밀도를 낮추면 값이 미세하게 틀어지고 그게 출력에 영향을 준다. 특히 긴 컨텍스트에서 "앞에 나온 특정 정보를 정확히 찾아오는" 작업에서 저하가 나타나기 쉽다. 일반적으로 FP8은 실무에서 널리 쓰이지만 INT4 이하는 신중하게 접근한다.

### 3-5. 순서를 바꾼다: Chunked Prefill

지금까지는 "캐시를 작게 만드는" 이야기였다. 방향이 조금 다른 기법이 하나 더 있는데, 1장에서 본 prefill/decode의 성격 차이를 스케줄링으로 푸는 방식이다.

문제 상황은 이렇다. 여러 사용자를 한 batch로 묶어 처리하는 중에 프롬프트가 7,000토큰짜리인 새 요청이 들어왔다고 하자. 이 요청의 prefill을 처리하는 동안 GPU가 그 작업에 붙잡히고, 이미 응답을 받고 있던 다른 사용자들의 decode는 그동안 멈춘다. 화면에서 글자가 흘러나오다가 갑자기 몇백 밀리초 멈추는 현상이 이래서 생긴다.

Sarathi가 제안한 **chunked prefill**은 이 긴 prefill을 작은 조각으로 쪼갠다. prefill 요청을 균등한 크기의 chunk로 나누고, batch를 구성할 때 prefill chunk 하나만 넣고 나머지 자리는 decode로 채우는 방식이다([arXiv:2308.16369](https://arxiv.org/abs/2308.16369)). 후속 논문인 Sarathi-Serve는 이걸 진행 중인 decode를 멈추지 않고 새 요청을 batch에 넣는 "stall-free 스케줄"이라고 표현한다(OSDI 2024).

그런데 여기서 이 글의 주제와 직접 맞물리는 대목이 나온다. **공짜가 아니고, 그 비용을 KV Cache가 치른다.** 논문은 이렇게 지적한다. 각 chunk는 같은 프롬프트의 앞선 모든 chunk의 KV Cache를 참조해야 하므로, 연산량은 그대로인데 GPU HBM에서 읽어오는 양이 늘어난다. prefill을 N개 chunk로 쪼개면 첫 번째 chunk의 KV Cache는 N-1번, 두 번째는 N-2번 다시 읽히는 식이다.

즉 chunked prefill은 지연시간을 얻는 대신 **메모리 대역폭을 더 쓴다**. 2장에서 계산한 42GB짜리 캐시를 떠올리면 이걸 여러 번 다시 읽는다는 게 어떤 부담인지 감이 온다. 여기서도 결국 KV Cache가 비용의 단위였다.

### 지표와의 관계 정리

성능 지표 관점에서 각 기법이 뭘 건드리는지 정리하면 이렇다.

| 기법 | 결정 시점 | 주로 개선되는 지표 | 주의할 점 |
|---|---|---|---|
| GQA / MLA | 학습 시 (모델 선택) | Throughput, 동시 요청 수 | 배포 시 변경 불가 |
| PagedAttention | 배포 시 | Throughput (batch 확대) | 커널 복잡도, 약간의 오버헤드 |
| Prefix caching | 배포 시 | **TTFT**, cache hit rate | 앞부분이 겹쳐야 효과 |
| Chunked prefill | 배포 시 | **TPOT/ITL** (끊김 완화) | KV Cache 재읽기로 대역폭 소모 증가 |
| KV quantization | 배포 시 | 동시 요청 수 | 품질 저하 가능 |

**여기까지 정리하면**: 구조 개선(GQA/MLA)은 모델을 고를 때 결정되고, 메모리 관리와 재사용, quantization은 배포 시 선택할 수 있다. 그런데 이걸 다 해도 GPU 메모리 자체가 늘어나는 건 아니다.

---

## 4. 그래도 부족할 때: KV Cache Offloading

### 문제 재정의

3장의 기법을 다 적용해도 근본적인 한계가 남는다. GPU HBM은 물리적으로 정해진 크기고 캐시는 계속 쌓인다. 결국 어느 시점에 오래된 캐시를 **버려야(eviction)** 한다.

버리면 어떻게 될까. 그 사용자가 다음 턴에 다시 말을 걸면 버린 캐시를 **처음부터 다시 계산**해야 한다. prefill을 다시 돌리는 것이다. 128K 컨텍스트라면 128K 토큰어치를 다시 계산한다.

WEKA는 이 상황을 이렇게 설명한다. HBM은 대단히 빠르지만 용량이 제한적이고 시스템 DRAM은 공간은 넉넉하지만 훨씬 느린데, 두 계층이 모두 차면 KV Cache가 축출되고 GPU는 이미 처리했던 토큰을 다시 계산하도록 강제되어 사이클과 전력과 시간을 낭비한다.

그래서 나온 아이디어가 단순하다. **버리지 말고 아래 계층에 내려놓자.**

### 메모리 계층

NVIDIA는 이 계층을 G1~G4로 이름 붙여 정리했다. NAND Research가 정리한 내용을 기준으로 보면 이렇다.

| 계층 | 위치 | 지연시간 | 특징 |
|---|---|---|---|
| G1 | GPU HBM | 나노초급 | 토큰 생성에 직접 쓰이는 활성 캐시. 가장 빠르지만 용량은 GPU 패키지 안에 든 만큼으로 제한 |
| G2 | 시스템 DRAM | 10~100나노초 | HBM에서 밀려난 캐시의 staging 공간. 단일 노드 메모리 용량에 묶임 |
| G3 | 로컬 SSD | 마이크로초급 | 짧은 주기 재사용용 warm 캐시. 노드 로컬이라 노드 간 공유가 안 됨 |
| G4 | 공유 스토리지 | 밀리초급 | 용량은 사실상 무제한, 노드 간 공유 가능. 원래 내구성 있는 기업 데이터용으로 설계됨 |

한 계층 내려갈 때마다 용량은 10~100배 커지고 속도는 그만큼 느려진다.

책상에 비유하면 G1은 지금 펼쳐놓은 서류, G2는 책상 서랍, G3은 사무실 캐비닛, G4는 지하 창고다. 창고까지 가면 오래 걸리지만 서류를 처음부터 다시 작성하는 것보다는 빠를 수 있다. "빠를 수 있다"가 핵심이고, 이걸 판단하는 게 다음 절이다.

### 핵심: 다시 계산할 것인가, 가져올 것인가

이 글에서 가장 중요한 부분이다. offloading이 이득인지 아닌지는 딱 하나의 비교로 결정된다.

```
가져오는 시간  <  다시 계산하는 시간   →   offloading이 이득
가져오는 시간  >  다시 계산하는 시간   →   그냥 다시 계산하는 게 낫다
```

양변을 풀어 쓰면 이렇다.

```
가져오는 시간 = KV Cache 바이트 수 ÷ 실제로 나오는 전송 속도
다시 계산하는 시간 = prefill 연산량 ÷ 실제로 나오는 연산 성능
```

Llama 3 70B, 128K 컨텍스트로 실제 숫자를 넣어보자. **아래는 내가 공개 스펙으로 한 어림 계산이고 벤더 발표 수치가 아니다.**

**가져오는 시간.** 2장에서 계산한 대로 KV Cache는 42GB다. WEKA는 자사 Augmented Memory Grid가 노드당 최대 252GB/s로 KV Cache를 전달한다고 발표했으니 이 값을 그대로 쓰면:

```
42 GB ÷ 252 GB/s ≈ 0.17 초
```

**다시 계산하는 시간.** prefill 연산량은 대략 `2 × 파라미터 수 × 토큰 수`로 어림잡는다.

```
2 × 70e9 × 131,072 ≈ 1.8 × 10^16 FLOP
```

H100/H200의 FP8 연산 성능은 약 1,979 TFLOPS다. (제조사 자료에는 3,958 TFLOPS로 적힌 경우가 많은데, 그건 sparsity라는 별도 기법을 적용했을 때의 값이라 일반적인 모델에는 해당되지 않는다.) 그런데 이 1,979도 이론상 최대치라 실제로는 다 못 쓴다. 카탈로그 속도와 실제 주행 연비가 다른 것과 같다. 실제로 쓰는 비율을 40%로 잡으면 약 790 TFLOPS다. (이 40%는 내가 임의로 잡은 값이고, 실제로는 구현과 batch 구성에 따라 크게 달라진다.)

```
1.8 × 10^16 ÷ 7.9 × 10^14 ≈ 23 초  (GPU 1장)
8장으로 나누면 ≈ 3 초
```

**비교.** 0.17초 대 3초. 20배 가까이 차이가 난다.

이 어림 계산이 방향은 맞는다는 걸 보여주는 벤더 수치가 있다. WEKA는 128,000 토큰을 처리할 때 prefill을 다시 계산하는 것 대비 TTFT가 20배 빨라졌다고 발표했다. 내 계산의 배수와 우연히 비슷한데 조건이 완전히 같은지는 확인할 수 없으니 "자릿수가 맞는다" 정도로만 받아들이는 게 맞겠다.

여기서 얻어야 할 감각은 이거다. **KV Cache는 대역폭만 충분하면 다시 계산하는 것보다 가져오는 게 압도적으로 싸다.** 컨텍스트가 길수록 유리해진다. 다시 계산하는 비용은 토큰 수에 비례해 늘지만, 가져오는 비용도 토큰 수에 비례하는데 계수가 훨씬 작기 때문이다.

반대로 **짧은 컨텍스트에서는 offloading이 손해**다. 4K 컨텍스트면 캐시가 1.3GB고 prefill도 몇십 밀리초면 끝난다. 왕복 지연만으로도 이득이 사라진다. offloading은 만능이 아니라 **긴 컨텍스트 전용 도구**다.

### 언제 쓰는가

이 손익 구조를 보면 어떤 워크로드에서 offloading이 빛나는지가 자연스럽게 나온다. 공통점은 "길고, 반복되고, 재사용된다"이다.

- **multi-turn 대화**: 턴이 쌓일수록 앞선 대화가 전부 컨텍스트에 남는다. 사용자가 커피 한 잔 마시고 오는 동안 캐시가 축출되고, 돌아와서 한마디 하면 전체를 다시 계산한다.
- **긴 system prompt**: 수천 토큰짜리 지침을 모든 요청이 공유한다. 한 번 계산해두면 계속 재사용된다.
- **RAG**: 같은 문서를 놓고 여러 질문이 들어온다.
- **agentic 워크플로우**: 에이전트는 도구를 호출하고 결과를 받아 다시 생각하는 과정을 반복하는데, 매 단계마다 그동안의 전체 히스토리가 컨텍스트에 들어간다. 사람이 한 번 질문할 때 모델은 열 번 이상 호출된다. 캐시 재사용이 없으면 같은 앞부분을 열 번 다시 계산한다. WEKA가 CoreWeave에서 한 테스트도 실제 agentic 워크로드를 모사한 시나리오를 대상으로 했다.
- **prefill/decode disaggregation**: 1장에서 봤듯 두 단계는 병목이 다르다. 그러면 아예 GPU를 나눠서 한쪽은 prefill만, 한쪽은 decode만 시키자는 발상이 나온다. 이렇게 하면 prefill 결과인 KV Cache를 **다른 GPU로 전송해야 한다**. offloading과 같은 데이터 경로를 쓴다.

바로 앞 3-5절의 chunked prefill과 비교하면 방향이 정반대라는 점이 흥미롭다. chunked prefill은 두 단계를 **더 잘 섞으려는** 시도고, disaggregation은 아예 **떼어놓으려는** 시도다. 한 서베이는 이 둘을 이렇게 대비한다. Orca가 iteration 단위 continuous batching을 도입하고 Sarathi-Serve가 chunked prefill로 부분 prefill을 진행 중인 decode batch에 접어 넣는 방식은 throughput은 높지만 prefill이 같은 자리의 decode를 막아 지연시간이 들쭉날쭉해진다. 두 단계의 요구가 상충한다는 관찰에서 출발해, DistServe와 Mooncake는 두 단계를 별도 인스턴스에서 돌리고 KV Cache 전송으로 잇는다.

이 계보의 출발점이 Splitwise다. prefill과 decode 단계를 서로 다른 머신에 나눠 배치해 throughput을 높이고, 단계별로 하드웨어를 다르게 최적화할 수 있게 하는 스케줄링 기법을 제안했다([arXiv:2311.18677](https://arxiv.org/abs/2311.18677)). "단계별로 하드웨어를 다르게"라는 부분이 핵심인데, prefill은 연산이 빠른 GPU가, decode는 메모리 대역폭이 큰 GPU가 유리하기 때문이다.

그리고 이 구조에서 KV Cache 전송이 어떤 위치에 놓이는지가 중요하다. 한 논문은 이들 시스템 모두가 KV Cache 전송을 disaggregation의 주된 오버헤드로 인정한다고 정리한다. DistServe는 같은 노드 배치를 선호해 이를 완화하고, Splitwise는 전송을 연산과 겹치게 하며, Mooncake는 저장·검색 경로를 최적화한다. 전송이 TTFT의 임계 경로 위에 있기 때문이다. 이 절 앞부분에서 계산한 대역폭 손익이 여기서도 그대로 적용된다.

Mooncake가 이 분리 구조의 대표 사례다. Moonshot AI의 Kimi 서비스를 돌리는 플랫폼인데, prefill과 decode 클러스터를 분리하고 GPU 클러스터에서 놀고 있는 CPU·DRAM·SSD 자원을 끌어모아 분산 KV Cache 풀을 만든다. 논문은 특정 시뮬레이션 시나리오에서 baseline 대비 throughput이 최대 525% 증가했다고 보고한다([arXiv:2407.00079](https://arxiv.org/abs/2407.00079), FAST 2025 Best Paper).

### 어떻게 옮기는가 — 데이터 경로

42GB를 0.17초에 옮기려면 평범한 파일 I/O로는 안 된다. 몇 가지 기술이 겹쳐서 동작한다.

**GPUDirect Storage (GDS)**: 원래 스토리지에서 GPU로 데이터를 보내려면 스토리지 → CPU 메모리 → GPU 메모리 순서로 두 번 복사해야 한다. GDS는 CPU 메모리를 거치지 않고 스토리지에서 GPU 메모리로 직접 보낸다. WEKA는 이 경로에 대해 NVMe와 GPU HBM 사이를 RDMA와 GPUDirect Storage로 직접 스트리밍해서 CPU와 DRAM 병목을 우회하는 마이크로초급 데이터 경로를 만든다고 설명한다.

**RDMA**: 원격 노드의 메모리를 상대 CPU 개입 없이 직접 읽고 쓰는 기술. VAST Data는 NFS-over-TCP로도 클라이언트 I/O를 채울 수 있지만, 지연시간과 호스트 CPU 사용률 최적화 그리고 GPUDirect Storage 활성화를 위해 RoCE 마운트를 권장한다고 문서에 적어뒀다.

**NIXL (NVIDIA Inference Transfer Library)**: 추론 워크로드 전용 전송 라이브러리다. GPU HBM, CPU DRAM, 로컬 SSD, 네트워크 스토리지 사이의 전송을 최적화하고, UCX·GDS·S3 같은 백엔드를 상황에 따라 자동 선택한다.

여기서 NIXL이 왜 중요한지 짚고 갈 필요가 있다. 이게 사실상 **스토리지 벤더와 추론 프레임워크를 잇는 표준 인터페이스** 역할을 한다. 벤더 입장에서는 NIXL 플러그인 하나만 만들면 vLLM, SGLang, TensorRT-LLM, Dynamo에 한꺼번에 붙는다. 5장의 벤더 비교표를 읽을 때 "NIXL 플러그인이 있는가"가 중요한 기준이 되는 이유다.

**소프트웨어 계층**은 대략 이렇게 나뉜다.

- **NVIDIA Dynamo**: 분산 추론 프레임워크. KV Cache offloading(GPU HBM → CPU DRAM → 로컬 SSD → 네트워크 스토리지), 기존 캐시가 있는 노드로 요청을 보내는 KV-aware 라우팅, NIXL을 통한 저지연 전송, vLLM·TensorRT-LLM 등과의 연동을 제공한다.
- **LMCache**: vLLM의 prefix cache에 영구 저장 백엔드를 붙이는 계층. vLLM의 `--swap-space`가 서버 인스턴스 하나의 CPU RAM에만 버퍼링하는 것과 달리, LMCache는 서버 재시작을 넘어 블록을 유지하고 여러 서버 replica 간에 공유한다.
- **Mooncake Store**: 위에서 본 분산 KV Cache 풀. 2025년 4월 LMCache가 Mooncake Store를 remote connector로 공식 지원하기 시작했고, SGLang도 Mooncake Transfer Engine을 지원한다.

### 새로 등장한 계층: NVIDIA CMX (G3.5)

2026년 들어 이 영역에서 큰 변화가 있었다.

NVIDIA는 CES 2026에서 ICMSP(Inference Context Memory Storage Platform)를 발표했다. GPU KV Cache를 NVMe 기반 스토리지로 확장하고, NVMe에 있는 KV Cache를 context memory 주소 공간의 일부로 만들어 추론 실행 간에 유지되도록 하는 것이 목표다. 이후 GTC 2026에서 이 이름이 CMX로 바뀌었고 STX 레퍼런스 아키텍처가 함께 발표됐다.

핵심은 **G3.5라는 계층을 새로 만든 것**이다. CMX는 STX 아키텍처의 첫 랙 스케일 구현으로, 노드 로컬 스토리지(G3)와 공유 엔터프라이즈 스토리지(G4) 사이에 새로운 G3.5 계층을 만든다. 이더넷 연결 플래시로 구성되며 pod 단위로 동작해서, pod 안의 모든 노드가 KV Cache를 공유해서 쓸 수 있다.

왜 이런 계층을 새로 만들었을까. 기존 G4의 설계 철학이 KV Cache와 안 맞기 때문이다. KV Cache는 성능에는 필수지만 본질적으로 임시적이고 다시 계산할 수 있는 데이터라, 복제·정합성 검사·메타데이터 관리 같은 전통적인 엔터프라이즈 스토리지 기능은 불필요한 오버헤드가 된다.

이 지적이 개인적으로 인상적이었다. 스토리지 업계가 수십 년간 쌓아온 "데이터를 절대 잃지 않는다"는 가치가 여기서는 오히려 짐이 된다는 뜻이다. 잃어버려도 다시 계산하면 되는 데이터니까.

그럼 어떻게 동작할까. 핵심은 **DOCA Memos**라는 소프트웨어 계층이다. NVIDIA는 이걸 BlueField-4와 CMX에 최적화된 SDK로, KV Cache를 AI 연산 노드와 CMX 데이터 노드 사이에서 관리·공유하며 단순한 key-value API를 제공해 이더넷에 붙은 플래시를 pod 단위 캐시 계층으로 바꾼다고 설명한다([NVIDIA CMX 제품 페이지](https://www.nvidia.com/en-us/data-center/ai-storage/cmx/)).

여기서 "key-value API"라는 표현이 눈에 띈다. 추론 프레임워크 입장에서는 파일을 열고 읽는 게 아니라 캐시에 키를 넣고 빼는 것처럼 보인다는 뜻이다. NVIDIA 개발자 블로그는 DOCA Memos가 context cache를 KV 관리·공유·배치의 일급 자원으로 다루는 통신·저장 계층이며, 추론 프레임워크와 인터페이스하고 BlueField-4가 실제 플래시 매체와의 전송을 담당한다고 설명한다. 그리고 이 방식이 stateless하게 확장되며 노드 간 공유를 위해 NIXL과 Dynamo를 활용한다고 덧붙인다([NVIDIA 개발자 블로그](https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/)).

BlueField-4가 하는 일도 명확하다. 같은 글에 따르면 Rubin 연산 노드의 DPU와 CMX 스토리지 트레이의 컨트롤러 양쪽에서 KV I/O와 control plane 동작을 가속해, 호스트 CPU 의존도와 호스트 메모리 복사를 줄인다. 4장 앞부분에서 본 GDS의 발상, 즉 "CPU를 데이터 경로에서 빼낸다"를 하드웨어 수준으로 밀어붙인 것으로 읽힌다.

NVIDIA는 CMX가 기존 스토리지 대비 최대 5배의 전력 효율과 최대 5배 높은 초당 토큰 수를 제공한다고 밝혔다. 다만 측정 조건은 공개된 자료에서 확인하지 못했다. 2026년 하반기 출시 예정이다.

### CMX를 얼마나 진지하게 받아들여야 할까

여기서 균형을 좀 잡아야 할 것 같다. 자료를 읽다 보면 이미 다 정해진 것처럼 느껴지는데, 비판적인 관측도 있다.

Blocks & Files는 CMX/STX가 너무 새로워서 대부분의 스토리지 공급업체가 지원을 계획 중이라고는 하지만 실제 제품 가용성은 아직 없다고 지적한다. 그리고 CMX와 STX가 NVIDIA의 Rubin GPU 랙에 적용되는 것이라, 현재로서는 그 랙이 들어간 대형 데이터센터 바깥의 추론 시장에는 해당되지 않는다고도 덧붙인다.

벤더들의 "지원 발표"를 어떻게 읽을지도 문제다. 한 분석가는 IBM이 Storage Scale과 통합하면 BlueField-4가 KV Cache의 효율적인 공유를 제공한다고 밝혔지만 그 통합이 어떻게 이뤄지는지에 대한 설명이 없다고 지적하고, Nutanix의 발표는 API 호환성 선언처럼 들린다고 평가한다. WEKA에 대해서도 주제적 연결은 있지만 BlueField-4와의 통합은 제시되지 않았다고 본다([Glenn K. Lockwood의 정리](https://glennklockwood.com/garden/ICMS)).

그러니까 5장의 벤더 목록을 볼 때 "CMX 파트너로 발표됨"과 "실제로 동작하는 통합이 있음"은 전혀 다른 이야기다. 이건 지금 이 시장 전체에 해당되는 이야기이기도 하다.

이 발표가 시장에 준 파장도 있다. Blocks & Files는 VAST Data 같은 업체는 길이 쉬워진 반면, GPU 서버 안의 로컬 SSD에 베팅했던 Hammerspace나 WEKA 같은 업체는 따라잡히거나 우회당할 수 있다고 분석했다. 이 부분은 벤더 간 이해관계가 얽힌 관측이라 그대로 받아들이기보다는 참고 정도로 두는 게 좋겠다.

**여기까지 정리하면**: offloading의 성립 조건은 "가져오는 시간 < 다시 계산하는 시간"이고 긴 컨텍스트일수록 유리하다. 데이터 경로는 GDS + RDMA + NIXL로 구성되며, NVIDIA는 이를 위해 G3.5라는 새 계층까지 만들고 있다.

---

## 5. 지원 벤더 정리

이제 벤더를 볼 차례다. 그런데 이 영역은 제품명이 자주 바뀌고 발표와 실제 출시 사이 간격도 커서 표를 그냥 던지면 오해하기 쉽다. **어떤 관점으로 읽어야 하는지 먼저 정리한다.**

읽는 법 세 가지:

1. **두 계층을 구분한다.** 스토리지 벤더(데이터를 담는 곳)와 프레임워크/미들웨어(데이터를 옮기고 관리하는 소프트웨어)는 경쟁 관계가 아니라 위아래 관계다. 실제 구축은 둘을 조합한다.
2. **연동 방식을 본다.** 앞서 말했듯 NIXL 플러그인이 있으면 여러 프레임워크에 한꺼번에 붙는다. 전용 플러그인만 있으면 특정 스택에 묶인다.
3. **수치는 벤더 발표값이다.** 아래 성능 수치는 전부 각 벤더가 자기 환경에서 측정한 값이다. 측정 조건이 공개된 것만 조건을 병기했고, 조건을 못 찾은 건 그렇게 표시했다. 벤더 간 직접 비교는 불가능하다고 보는 게 맞다.
4. **"지원 발표"와 "동작하는 통합"을 구분한다.** 4장 마지막에서 봤듯이, 파트너 명단에 이름이 올라간 것과 실제로 검증된 통합이 있는 것은 다르다. 아래 표에서 출처가 자사 기술 블로그이고 구체적인 연동 방식과 수치가 함께 있는 항목이 상대적으로 실체가 있는 쪽이다.

### (A) 스토리지 벤더

| 벤더 | 제품/기능명 | 연동 방식 | 공개 수치 (측정 조건) | 출처 |
|---|---|---|---|---|
| WEKA | Augmented Memory Grid (NeuralMesh 기반) | GDS + RDMA, 오픈소스 NIXL 플러그인, Dynamo/TensorRT-LLM/LMCache | 노드당 최대 252GB/s. 128K 토큰에서 재계산 대비 TTFT 20배 (OCI 공동 테스트). CoreWeave 8노드×8 H100(64 GPU) 프로덕션 테스트에서 TTFT 최대 6배 단축, GPU당 토큰 최대 4.2배 (장문 컨텍스트 테스트 기준) | [제품 페이지](https://www.weka.io/product/augmented-memory-grid/), [NIXL 연동 블로그](https://www.weka.io/blog/ai-ml/weka-accelerates-ai-inference-with-nvidia-dynamo-and-nvidia-nixl/) |
| VAST Data | VUA (VAST Undivided Attention), 오픈소스 | NIXL + GDS, NFS multipathing, RoCE 권장 | 장문 컨텍스트 시나리오에서 prefill 시간 10배 개선 (NIXL + GDS + vLLM/LMCache 연동 자체 실험) | [Dynamo 연동 블로그](https://www.vastdata.com/blog/nvidia-dynamo-vast-scalable-optimized-inference) |
| DDN | Infinia | NIXL 플러그인이 NIXL 1.3부터 공식 패키지 및 Dynamo 컨테이너에 기본 포함 | 서브 밀리초 지연 주장. 구체적 측정 조건 미확인 | [DDN 블로그](https://www.ddn.com/blog/ddn-becomes-the-first-storage-vendor-natively-integrated-into-nvidia-kv-cache-management/) |
| Hammerspace | Tier Zero | GPU 서버 로컬 NVMe 활용 방식 | 미확인 | [NAND Research](https://nand-research.com/research-note-improving-inference-nvidias-inference-context-memory-storage-platform/) |
| Dell | PowerScale / ObjectScale / Project Lightning(프라이빗 프리뷰) | LMCache + NIXL 통합 | 미확인 | 동일 |
| Pure Storage, IBM, NetApp, HPE, Hitachi Vantara, Cloudian, Nutanix, Supermicro, AIC | — | NVIDIA ICMSP/CMX 초기 파트너로 발표됨 | 미확인 (BlueField-4 기반, 2026년 하반기 예정) | [Blocks & Files](https://blocksandfiles.com/2026/01/06/nvidia-standardizes-gpu-cluster-kv-cache-offload-to-nvme-ssds/) |

DDN 사례는 흐름을 잘 보여준다. 7월 NIXL 1.3부터 Infinia 지원이 NVIDIA 공식 NIXL 패키지와 사전 빌드된 Dynamo 추론 컨테이너에 기본 탑재되어, NIXL을 설치하거나 Dynamo 컨테이너를 받으면 DDN 클라이언트 패키지만 추가한 뒤 코드 변경 없이 설정만으로 KV Cache 가속을 켤 수 있다. 벤더별 커스텀 통합에서 표준 플러그인 방식으로 넘어가는 중이라는 신호다.

### (B) 프레임워크 / 미들웨어

| 이름 | 성격 | 지원 엔진 | 특징 |
|---|---|---|---|
| NVIDIA Dynamo | 분산 추론 프레임워크 | vLLM, TensorRT-LLM, SGLang | KV Block Manager, KV-aware 라우팅, NIXL 전송 |
| NIXL | 전송 라이브러리 | Dynamo, vLLM, LMCache, TensorRT-LLM, SGLang | 사실상 표준 인터페이스. `pip install nixl` |
| LMCache | KV Cache 저장 계층 | vLLM, SGLang, Dynamo | 서버 재시작 넘어 유지, replica 간 공유 |
| Mooncake | 분산 KV Cache 풀 + P/D 분리 | vLLM, SGLang, LMCache 연동 | FAST 2025 Best Paper, Kimi 프로덕션 |
| vLLM / SGLang | 추론 엔진 | — | PagedAttention / RadixAttention 원조 |

### 직접 구축할 것인가, 도입할 것인가

지금 시점에서 선택지를 정리하면 대략 세 가지다.

**직접 구축**: LMCache + 로컬 NVMe 조합이 진입 장벽이 가장 낮다. 오픈소스고, vLLM을 쓰고 있다면 붙이기 어렵지 않다. 노드 로컬 재사용까지만 필요하면 이걸로 충분할 수 있다. 대신 노드 간 공유, 용량 확장, 운영 부담은 직접 감당해야 한다.

**벤더 솔루션 도입**: 노드를 넘어 KV Cache를 공유해야 하거나, 페타바이트급 용량이 필요하거나, cache hit rate를 SLO 수준으로 관리해야 하면 스토리지 벤더 제품이 답이 된다. 다만 위 표의 성능 수치는 각자 유리한 조건에서 낸 값이므로 도입 전에 자기 워크로드로 PoC를 하는 게 필수라고 본다. 특히 **평균 컨텍스트 길이**와 **캐시 재사용률**을 먼저 측정해야 한다. 4장에서 봤듯 짧은 컨텍스트에서는 offloading이 손해다.

**기다리기**: CMX가 2026년 하반기에 나온다. 지금 대규모 투자를 하면 몇 달 뒤에 표준이 바뀔 수 있다. 반대로 마냥 기다리면 그동안의 GPU 낭비를 감수해야 한다.

솔직히 나라면 지금은 "측정부터 한다"에 가까울 것 같다. 우리 워크로드의 평균 컨텍스트 길이가 4K인지 128K인지에 따라 답이 정반대로 나오는데, 그 숫자를 모르고 벤더 비교표부터 보는 건 순서가 틀렸다는 생각이 든다.

---

## 6. 2주차 과제를 하면서 든 생각

1주차에 microGPT를 보면서 "KV Cache라는 게 결국 익숙한 캐싱 개념이구나" 하고 생각했는데, 2주차에 파고들어 보니 그 안심이 좀 성급했다. 개념은 익숙한 게 맞지만, 그 익숙한 개념이 42GB라는 크기를 만나는 순간 완전히 다른 문제가 된다는 걸 이번에 알았다.

가장 크게 바뀐 인식은 이거다. 그동안 "AI 인프라"라고 하면 GPU 성능을 먼저 떠올렸는데, 추론 시장이 성장할수록 **GPU가 모자란 부분을 스토리지가 어떻게 메울 것인가**가 화두였다. 스토리지 회사들이 추론 시장에 뛰어들고 NVIDIA가 새 계층까지 만드는 이유가 여기 있었다.

---

## 11줄 요약

1. KV Cache는 사실상 추론 전용이다. 가중치가 고정돼 있고 토큰을 하나씩 만들 때만 의미가 있어서, 전체 시퀀스를 병렬로 처리하는 학습에는 등장하지 않는다.
2. 앞선 토큰의 Key/Value를 재계산하지 않기 위한 필수 장치이며, 없으면 연산량이 토큰 수의 세제곱에 비례한다.
3. 캐시를 쓰면 추론이 prefill(compute-bound)과 decode(memory-bandwidth-bound)로 나뉘고, TTFT와 TPOT라는 서로 다른 지표가 여기서 갈라진다.
4. 토큰당 KV Cache = layer 수 × 2 × KV head 수 × head dim × 데이터 타입 바이트.
5. Llama 3 70B(FP16) 기준 토큰당 320KB, 128K 컨텍스트면 요청 하나에 약 42GB. H200 한 장으로 동시 처리 가능한 요청이 두 개가 안 된다.
6. 구조 개선(MQA/GQA/MLA)은 캐시 크기를 줄이지만 학습 시점에 결정되므로 모델 선택 문제다.
7. PagedAttention은 OS의 virtual memory를 본떠 fragmentation을 없앴고, 논문에서 기존 시스템 대비 throughput 2~4배를 보고했다. 지금은 업계 표준이다.
8. Prefix caching은 앞부분이 겹치는 요청에서 TTFT를 크게 줄이고, chunked prefill은 끊김을 줄이는 대신 KV Cache를 여러 번 다시 읽어 대역폭을 더 쓴다.
9. offloading의 성립 조건은 "가져오는 시간 < 다시 계산하는 시간"이다. 긴 컨텍스트에서 유리하고 짧은 컨텍스트에서는 손해다.
10. 데이터 경로는 GPUDirect Storage + RDMA + NIXL로 구성되며, NIXL이 스토리지 벤더와 추론 프레임워크를 잇는 사실상 표준이 됐다.
11. WEKA, VAST, DDN, Hammerspace, Dell 등이 제품을 내놨고 NVIDIA는 CMX로 G3.5 계층을 신설했지만, 파트너 명단에 이름이 올라간 것과 실제 동작하는 통합이 있는 것은 아직 다르다.

---

## 용어 정리 (부록)

1주차에서 이미 정리한 용어(토큰, 어텐션, Q/K/V, KV Cache, softmax, 파라미터 등)는 [1주차 글의 용어 정리](https://youwins-lab.github.io/Blog/study/llm-serving/2026/08/08/microgpt-week1.html)를 참고. 여기서는 이번 주에 새로 나온 것만 정리한다.

**추론 단계와 지표**

- **autoregressive 생성**: 토큰을 한 번에 하나씩, 앞의 결과를 보고 다음을 만드는 방식.
- **prefill**: 입력 프롬프트 전체를 한 번에 처리해 KV Cache를 채우는 단계.
- **decode**: 그 뒤 토큰을 하나씩 생성하는 단계.
- **compute-bound / memory-bandwidth-bound**: 연산 능력이 부족해 느린 상태 / 메모리에서 데이터 읽어오는 속도가 부족해 느린 상태.
- **arithmetic intensity**: 메모리에서 1바이트를 읽을 때 수행하는 연산 횟수. 높으면 compute-bound, 낮으면 memory-bandwidth-bound.
- **TTFT (Time To First Token)**: 요청 후 첫 토큰이 나올 때까지의 시간.
- **TPOT (Time Per Output Token) / ITL (Inter-Token Latency) / TBT (Time Between Tokens)**: 토큰 하나를 만드는 데 걸리는 시간. 논문마다 용어가 다르지만 거의 같은 개념이다.
- **throughput / goodput**: 초당 처리 토큰 수 / 그중 SLO를 지킨 것만 센 값.
- **SLO (Service Level Objective)**: 지켜야 할 응답 시간 등의 목표치.

**어텐션 변형**

- **MHA (Multi-Head Attention)**: 원조 방식. Query head마다 Key/Value head가 따로 있다.
- **MQA (Multi-Query Attention)**: 모든 Query head가 Key/Value 한 세트를 공유. 캐시는 가장 작지만 품질 저하가 있다.
- **GQA (Grouped-Query Attention)**: Query head를 그룹으로 묶어 그룹당 Key/Value 한 세트를 공유. MHA와 MQA의 절충.
- **MLA (Multi-head Latent Attention)**: Key/Value를 저차원 latent 벡터로 압축해 저장. DeepSeek-V2/V3에서 사용.
- **uptraining**: 이미 학습된 MHA 모델을 적은 추가 학습으로 GQA/MQA로 변환하는 기법.

**메모리 관리**

- **fragmentation**: 메모리가 남아 있는데도 조각나 있거나 예약에 묶여 있어 실제로는 못 쓰는 상태. internal(예약해놓고 안 씀)과 external(흩어져 있어 큰 덩어리를 못 만듦)로 나뉜다.
- **PagedAttention**: KV Cache를 고정 크기 블록으로 쪼개 비연속 저장하는 방식. OS의 paging을 본떴다. vLLM에서 처음 구현.
- **RadixAttention**: 요청 간 공유되는 앞부분을 radix tree로 관리해 재사용하는 방식. SGLang에서 구현.
- **prefix caching**: 여러 요청이 공유하는 앞부분의 KV Cache를 한 번만 계산해 재사용하는 것.
- **cache hit rate**: 요청 중 기존 캐시를 재사용할 수 있었던 비율.
- **eviction**: 공간이 부족할 때 오래된 캐시를 버리는 것.
- **continuous batching**: batch가 다 끝나기를 기다리지 않고 요청이 끝나는 대로 새 요청을 그 자리에 채우는 스케줄링. Orca에서 제안.
- **chunked prefill**: 긴 prefill을 작은 조각으로 쪼개 decode 사이에 끼워 넣어 진행 중인 응답이 끊기지 않게 하는 방식. Sarathi에서 제안.
- **quantization**: 값을 더 적은 비트로 표현해 메모리를 줄이는 것. FP16 → FP8 → INT4 순으로 작아진다.

**offloading과 데이터 경로**

- **offloading**: KV Cache를 GPU 메모리 밖(DRAM, SSD, 네트워크 스토리지)에 내려놓고 필요할 때 다시 가져오는 것.
- **G1~G4 (그리고 G3.5)**: NVIDIA가 정의한 메모리·스토리지 계층. GPU HBM / 시스템 DRAM / 로컬 SSD / 공유 스토리지, 그리고 그 사이에 새로 낀 이더넷 연결 플래시 계층.
- **GPUDirect Storage (GDS)**: CPU 메모리를 거치지 않고 스토리지에서 GPU 메모리로 직접 데이터를 보내는 기술.
- **RDMA**: 상대 CPU 개입 없이 원격 메모리를 직접 읽고 쓰는 기술.
- **NIXL (NVIDIA Inference Transfer Library)**: 추론용 데이터 전송 라이브러리. 스토리지 벤더와 추론 프레임워크를 잇는 표준 역할을 한다.
- **KVBM (KV Block Manager)**: Dynamo에서 KV 블록의 위치와 이동을 관리하는 구성 요소.
- **prefill/decode disaggregation**: prefill과 decode를 서로 다른 GPU 그룹에 나눠 맡기는 구조. 그 사이에 KV Cache 전송이 필요하다.
- **CMX (Context Memory eXtension)**: NVIDIA가 CES 2026에 ICMSP로 발표하고 GTC 2026에서 개명한 KV Cache 전용 스토리지 플랫폼. BlueField-4 기반이며 Rubin 플랫폼을 전제로 한다.
- **DOCA Memos**: CMX에서 KV Cache를 관리·공유하는 SDK. 추론 프레임워크에 key-value API 형태로 노출된다.
- **DPU (Data Processing Unit)**: 네트워크·스토리지 처리를 CPU에서 떼어내 전담하는 프로세서.
- **JBOF (Just a Bunch Of Flash)**: 컨트롤러 기능을 최소화하고 플래시 드라이브만 모아놓은 장비 형태.

---

## 참고 자료

**논문**

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023 — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (MQA) — [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (MLA) — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- Agrawal et al., *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills* — [arXiv:2308.16369](https://arxiv.org/abs/2308.16369)
- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, OSDI 2024 — [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)
- Patel et al., *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*, ISCA 2024 — [arXiv:2311.18677](https://arxiv.org/abs/2311.18677)
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (RadixAttention), NeurIPS 2024 — [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
- Qin et al., *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*, FAST 2025 Best Paper — [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)
- Prabhu et al., *vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention*, ASPLOS 2025 — [arXiv:2405.04437](https://arxiv.org/abs/2405.04437)
- *A Survey on Large Language Model Acceleration based on KV Cache Management* — [arXiv:2412.19442](https://arxiv.org/abs/2412.19442)
- *LLM Inference Serving: Survey of Recent Advances and Opportunities* — [arXiv:2407.12391](https://arxiv.org/abs/2407.12391)

**벤더 문서**

- WEKA, [Augmented Memory Grid 제품 페이지](https://www.weka.io/product/augmented-memory-grid/)
- WEKA, [NVIDIA Dynamo/NIXL 연동 블로그](https://www.weka.io/blog/ai-ml/weka-accelerates-ai-inference-with-nvidia-dynamo-and-nvidia-nixl/)
- WEKA, [CoreWeave 테스트 결과](https://www.weka.io/blog/ai-ml/agents-at-scale-escaping-upside-down-tokenomics-with-up-to-4-2x-efficiency/)
- WEKA, [What Is Context Memory?](https://www.weka.io/learn/ai-ml/what-is-context-memory-primary/)
- VAST Data, [NVIDIA Dynamo + VAST](https://www.vastdata.com/blog/nvidia-dynamo-vast-scalable-optimized-inference)
- VAST Data, [How NVIDIA Dynamo and VAST Unlock Context Reuse at Scale](https://www.vastdata.com/blog/how-nvidia-dynamo-vast-unlock-context-reuse-at-scale)
- DDN, [First Storage Vendor Natively Integrated into NVIDIA KV Cache Management](https://www.ddn.com/blog/ddn-becomes-the-first-storage-vendor-natively-integrated-into-nvidia-kv-cache-management/)
- Mooncake, [공식 문서](https://kvcache-ai.github.io/Mooncake/)
- NVIDIA, [CMX 제품 페이지](https://www.nvidia.com/en-us/data-center/ai-storage/cmx/)
- NVIDIA, [Introducing BlueField-4-Powered CMX (개발자 블로그)](https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/)
- NVIDIA, [TensorRT-LLM H200 스펙 문서](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/H200launch.md)

**분석 및 블로그**

- NAND Research, [NVIDIA STX & CMX 리서치 노트](https://nand-research.com/nvidia-stx-cmx-infrastructure-for-agentic-ai-context-storage/)
- NAND Research, [ICMSP 리서치 노트](https://nand-research.com/research-note-improving-inference-nvidias-inference-context-memory-storage-platform/)
- Blocks & Files, [NVIDIA standardizes KV cache offload to NVMe SSDs](https://blocksandfiles.com/2026/01/06/nvidia-standardizes-gpu-cluster-kv-cache-offload-to-nvme-ssds/)
- Blocks & Files, [NVIDIA and its partners' KV Cache extenders](https://www.blocksandfiles.com/ai-ml/2026/03/30/nvidia-and-its-partners-kv-cache-extenders/5209284)
- Glenn K. Lockwood, [NVIDIA CMX 정리](https://glennklockwood.com/garden/ICMS)
- JAX ML, [Scaling Book — Serving LLaMA 3-70B](https://jax-ml.github.io/scaling-book/applied-inference/)
- Red Hat Developer, [How PagedAttention resolves memory waste](https://developers.redhat.com/articles/2025/07/24/how-pagedattention-resolves-memory-waste-llm-systems)
