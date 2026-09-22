---
layout: post
title: "AI 칩 여섯 개를 같은 기준으로 읽어봤다 — NVIDIA, TPU, AMD, Cerebras, Trainium, Groq"
date: 2026-09-19
categories: study hardware ai-chip-architectures
tags: [NVIDIA, TPU, AMD, Cerebras, Trainium, Groq, GPU, Networking, LLMSO]
---

# AI 칩 여섯 개를 같은 기준으로 읽어봤다

NVLink, ICI, UALink, NeuronLink. 이름은 다 다른데 설명을 읽어보면 결국 비슷한 일을 한다. 칩도 마찬가지였다. TPU가 GPU랑 뭐가 다르냐고 물으면 "전용 칩이라 빠르다"에서 더 나가지 못했다.

Jacob Peake의 ["AI Chip Architectures"](https://www.jepeake.com/ai-chip-architectures)는 여섯 회사를 한 편에 넣고 같은 축으로 비교한다. LLMSO 마지막 과제로 이 글을 골라 읽었고, 읽으면서 정리한 게 아래 내용이다. 네트워크를 하다 보니 칩 내부보다 칩끼리 연결하는 쪽을 더 오래 봤다. 11장은 아예 원문에 없는 내 얘기다.

### 먼저 결론부터

- 여섯 회사가 푸는 문제는 같다. 계산 속도가 아니라 숫자를 계산기까지 옮기는 속도다. (2장)
- NVIDIA의 해자는 칩이 아니다. 20년치 남의 코드, 그리고 고객사 안에 상주하는 엔지니어다. (3장)
- TPU는 캐시도 스케줄러도 없앴다. 컴파일러가 틀리면 막아줄 게 없다. (4장)
- AMD는 메모리로 이기고 랙으로 졌다. 랙 단위 제품이 2026년에야 나온다. (5장)
- Cerebras는 속도가 진짜다. 대신 메모리가 44 GB에서 안 자란다. 공정이 좋아져도 안 는다. (6장)
- Trainium은 TPU 설계를 빌리고 AWS 네트워크를 재사용한다. 싸게 파는 게 전략이다. (7장)
- Groq는 불확실한 요소를 전부 없앴다. 대신 모델 하나에 칩이 수백 개 든다. (8장)
- 네트워크에서 제일 크게 바뀐 건 MoE다. 토폴로지 고르는 기준이 대역폭에서 홉 수로 옮겨갔다. (11장)

여섯 개를 다 읽고 나서 두 축으로 정리해봤다. 데이터를 어디에 두느냐, 그리고 실행 순서를 누가 정하느냐다.

<figure style="margin:36px 0">
<svg viewBox="0 0 740 400" role="img" aria-label="여섯 아키텍처를 메모리 위치와 스케줄링 주체 두 축에 놓은 지도" style="width:100%;height:auto;font-family:var(--sans,system-ui,sans-serif)">
<line x1="70" y1="340" x2="710" y2="340" stroke="currentColor" stroke-opacity="0.25" stroke-width="1"/>
<line x1="70" y1="40" x2="70" y2="340" stroke="currentColor" stroke-opacity="0.25" stroke-width="1"/>
<line x1="70" y1="190" x2="710" y2="190" stroke="currentColor" stroke-opacity="0.1" stroke-width="1" stroke-dasharray="4 4"/>
<line x1="390" y1="40" x2="390" y2="340" stroke="currentColor" stroke-opacity="0.1" stroke-width="1" stroke-dasharray="4 4"/>
<text x="70" y="366" font-size="12" fill="currentColor" fill-opacity="0.6">HBM 중심</text>
<text x="710" y="366" font-size="12" text-anchor="end" fill="currentColor" fill-opacity="0.6">칩 안 SRAM만</text>
<text x="390" y="386" font-size="12" text-anchor="middle" fill="currentColor" fill-opacity="0.45">데이터를 어디에 두나</text>
<text x="60" y="46" font-size="12" text-anchor="end" fill="currentColor" fill-opacity="0.6">하드웨어가</text>
<text x="60" y="62" font-size="12" text-anchor="end" fill="currentColor" fill-opacity="0.6">정함</text>
<text x="60" y="330" font-size="12" text-anchor="end" fill="currentColor" fill-opacity="0.6">컴파일러가</text>
<text x="60" y="346" font-size="12" text-anchor="end" fill="currentColor" fill-opacity="0.6">정함</text>
<circle cx="150" cy="80" r="6" fill="currentColor"/>
<text x="166" y="78" font-size="14" font-weight="600" fill="currentColor">NVIDIA</text>
<text x="166" y="95" font-size="12" fill="currentColor" fill-opacity="0.58">캐시 + 커널 코드</text>
<circle cx="120" cy="130" r="6" fill="currentColor"/>
<text x="136" y="128" font-size="14" font-weight="600" fill="currentColor">AMD</text>
<text x="136" y="145" font-size="12" fill="currentColor" fill-opacity="0.58">+ 256 MB 캐시</text>
<circle cx="230" cy="270" r="6" fill="currentColor"/>
<text x="246" y="268" font-size="14" font-weight="600" fill="currentColor">Google TPU</text>
<text x="246" y="285" font-size="12" fill="currentColor" fill-opacity="0.58">캐시 없음, XLA가 전부</text>
<circle cx="270" cy="306" r="6" fill="currentColor"/>
<text x="286" y="304" font-size="14" font-weight="600" fill="currentColor">AWS Trainium</text>
<text x="286" y="321" font-size="12" fill="currentColor" fill-opacity="0.58">TPU 방식을 빌려옴</text>
<circle cx="640" cy="175" r="6" fill="currentColor"/>
<text x="624" y="173" font-size="14" font-weight="600" text-anchor="end" fill="currentColor">Cerebras</text>
<text x="624" y="190" font-size="11.5" text-anchor="end" fill="currentColor" fill-opacity="0.55">데이터 도착 순서가 곧 스케줄</text>
<circle cx="672" cy="300" r="6" fill="currentColor"/>
<text x="656" y="298" font-size="14" font-weight="600" text-anchor="end" fill="currentColor">Groq</text>
<text x="656" y="315" font-size="11.5" text-anchor="end" fill="currentColor" fill-opacity="0.55">네트워크까지 컴파일러가</text>
<rect x="96" y="56" width="180" height="290" rx="12" fill="none" stroke="currentColor" stroke-opacity="0.18" stroke-width="1"/>
<rect x="470" y="150" width="240" height="176" rx="12" fill="none" stroke="currentColor" stroke-opacity="0.18" stroke-width="1"/>
<text x="186" y="46" font-size="11.5" text-anchor="middle" fill="currentColor" fill-opacity="0.4">HBM을 쓰는 네 개</text>
<text x="590" y="142" font-size="11.5" text-anchor="middle" fill="currentColor" fill-opacity="0.4">HBM을 안 쓰는 두 개</text>
</svg>
<figcaption style="font-size:13px;color:var(--text-muted);margin-top:12px;line-height:1.65">여섯 개를 두 축에 놓아봤다. 왼쪽 위(NVIDIA, AMD)와 왼쪽 아래(TPU, Trainium)가 갈라지고, HBM을 아예 안 쓰는 두 개가 오른쪽에 따로 선다. 원문에 있는 그림은 아니고 읽으면서 내가 정리한 것이다.</figcaption>
</figure>

## 1. 왜 이런 칩들이 생겼나

2018년에 Hennessy와 Patterson이 튜링상 강연을 했다. 1980년대엔 CPU 성능이 해마다 52%씩 올랐는데 2018년엔 3%였다. 이제는 한 가지 일만 잘하는 전용 칩으로 가야 한다는 얘기였고, 사례로 든 게 TPU 1세대였다. CPU보다 추론이 29배 빠르고 전력 효율은 80배. 강연 마지막 예측은 "앞으로 10년은 컴퓨터 구조의 캄브리아기 대폭발이 될 것"이었다.

예측은 맞았다. 다만 실제로 대규모로 깔린 건 네 부류로 추려진다. GPU(NVIDIA, AMD), 행렬 곱셈 전용 가속기(Google TPU, AWS Trainium), 웨이퍼를 통째로 쓰는 칩(Cerebras), 그리고 LPU(Groq)다.

2026년 현재 판세는 이렇다. NVIDIA가 확실한 선두고 AMD가 쫓아간다. OpenAI와 Meta가 각각 6 GW 규모로 계약했다. TPU는 Gemini를 학습시키고 Anthropic에 최대 100만 칩을 공급할 예정인데, 같은 Anthropic이 Trainium 100만 칩 위에서도 Claude를 돌린다. Cerebras는 OpenAI 추론을 서빙하고, Groq는 200억 달러 규모로 NVIDIA에 인수됐다.

## 2. 진짜 문제는 계산이 아니라 데이터 옮기기

Transformer는 행렬 곱셈의 연속이다. 중간에 정규화나 활성화 함수가 끼어들지만 계산량의 99%는 행렬 곱셈이고, 큰 모델 하나 학습시키는 데 곱셈-덧셈이 10의 25승 번쯤 들어간다.

문제는 그 곱셈의 모양이 상황마다 다르다는 것이다.

| 상황 | 하는 일 | 병목 |
| --- | --- | --- |
| 학습 | 행렬 × 행렬 | 계산이 바쁘다 |
| 첫 응답 준비(prefill) | 행렬 × 행렬, 입력 전체를 한 번에 | 계산이 바쁘다 |
| 토큰 한 개씩 뽑기(decode) | 행렬 × 벡터 | 메모리가 바쁘다 |

decode가 아픈 건 구조 때문이다. 앞 토큰이 나와야 다음 토큰을 시작할 수 있고, 한 번에 한 토큰만 처리하니 행렬끼리 곱하는 게 아니라 행렬에 벡터 하나를 곱하는 모양이 된다. 그런데 그 토큰 하나를 뽑으려고 모델 weight를 전부 한 번 읽는다. 지금까지 나온 문맥(KV cache)도 같이 읽는다.

<figure style="margin:36px 0">
<svg viewBox="0 0 740 286" role="img" aria-label="학습은 한 번 읽은 weight로 여러 토큰을 처리하고, 디코드는 토큰 하나마다 weight 전체를 다시 읽는다" style="width:100%;height:auto;font-family:var(--sans,system-ui,sans-serif)">
<defs>
<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
</marker>
<marker id="ar-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--accent,#4f46e5)"/>
</marker>
</defs>
<text x="20" y="24" font-size="13" font-weight="600" fill="currentColor">학습 · 첫 응답 준비</text>
<rect x="20" y="40" width="330" height="230" rx="10" fill="currentColor" fill-opacity="0.03" stroke="currentColor" stroke-opacity="0.15"/>
<rect x="42" y="70" width="80" height="130" rx="8" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-opacity="0.35"/>
<text x="82" y="128" font-size="12" text-anchor="middle" fill="currentColor">HBM</text>
<text x="82" y="146" font-size="11" text-anchor="middle" fill="currentColor" fill-opacity="0.55">weight 전체</text>
<line x1="126" y1="135" x2="186" y2="135" stroke="currentColor" stroke-width="2" marker-end="url(#ar)"/>
<text x="156" y="124" font-size="11" text-anchor="middle" fill="currentColor" fill-opacity="0.6">한 번</text>
<rect x="190" y="70" width="80" height="130" rx="8" fill="currentColor" fill-opacity="0.04" stroke="currentColor" stroke-opacity="0.35"/>
<rect x="190" y="88" width="80" height="112" rx="8" fill="currentColor" fill-opacity="0.16"/>
<text x="230" y="128" font-size="12" text-anchor="middle" fill="currentColor">곱셈기</text>
<text x="230" y="146" font-size="11" text-anchor="middle" fill="currentColor" fill-opacity="0.55">거의 꽉 참</text>
<line x1="274" y1="135" x2="322" y2="135" stroke="currentColor" stroke-width="2" marker-end="url(#ar)"/>
<text x="298" y="163" font-size="11" text-anchor="middle" fill="currentColor" fill-opacity="0.7">토큰</text>
<text x="298" y="178" font-size="11" text-anchor="middle" fill="currentColor" fill-opacity="0.7">수백 개</text>
<text x="185" y="252" font-size="12" text-anchor="middle" fill="currentColor" fill-opacity="0.6">읽은 바이트당 계산이 많다 → 계산이 병목</text>
<text x="390" y="24" font-size="13" font-weight="600" fill="var(--accent,#4f46e5)">토큰 하나씩 뽑기</text>
<rect x="390" y="40" width="330" height="230" rx="10" fill="var(--accent-soft,rgba(79,70,229,0.06))" stroke="var(--accent,#4f46e5)" stroke-opacity="0.35"/>
<rect x="412" y="70" width="80" height="130" rx="8" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-opacity="0.35"/>
<text x="452" y="128" font-size="12" text-anchor="middle" fill="currentColor">HBM</text>
<text x="452" y="146" font-size="11" text-anchor="middle" fill="currentColor" fill-opacity="0.55">weight 전체</text>
<line x1="496" y1="135" x2="556" y2="135" stroke="var(--accent,#4f46e5)" stroke-width="2" marker-end="url(#ar-a)"/>
<text x="526" y="124" font-size="11" text-anchor="middle" fill="var(--accent,#4f46e5)">매번</text>
<rect x="560" y="70" width="80" height="130" rx="8" fill="currentColor" fill-opacity="0.04" stroke="currentColor" stroke-opacity="0.35"/>
<rect x="560" y="182" width="80" height="18" rx="6" fill="currentColor" fill-opacity="0.16"/>
<text x="600" y="120" font-size="12" text-anchor="middle" fill="currentColor">곱셈기</text>
<text x="600" y="138" font-size="11" text-anchor="middle" fill="currentColor" fill-opacity="0.55">대부분</text>
<text x="600" y="153" font-size="11" text-anchor="middle" fill="currentColor" fill-opacity="0.55">기다린다</text>
<line x1="644" y1="135" x2="692" y2="135" stroke="currentColor" stroke-width="2" marker-end="url(#ar)"/>
<text x="668" y="163" font-size="11" text-anchor="middle" fill="currentColor" fill-opacity="0.7">토큰</text>
<text x="668" y="178" font-size="11" text-anchor="middle" fill="currentColor" fill-opacity="0.7">1개</text>
<path d="M 668 196 L 668 228 L 452 228 L 452 208" fill="none" stroke="var(--accent,#4f46e5)" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#ar-a)"/>
<text x="560" y="222" font-size="11.5" text-anchor="middle" fill="var(--accent,#4f46e5)">토큰마다 처음부터 다시</text>
<text x="555" y="252" font-size="12" text-anchor="middle" fill="currentColor" fill-opacity="0.6">읽은 바이트당 계산이 거의 없다 → 메모리가 병목</text>
</svg>
<figcaption style="font-size:13px;color:var(--text-muted);margin-top:12px;line-height:1.65">같은 칩, 같은 파이프다. 달라지는 건 한 번 읽어서 몇 개를 처리하느냐뿐이다. 토큰을 하나씩 뽑을 때는 weight 전체를 읽고 토큰 하나를 얻으니 곱셈기가 대부분 기다린다.</figcaption>
</figure>

서빙 시스템이 하는 일은 결국 이 모양을 억지로 되돌리는 것이다. 여러 사용자 요청을 같은 스텝에 쌓아 올리고(continuous batching), 초안 토큰을 여러 개 만들어 한 번에 검증하고(speculative decoding), 아예 같은 요령을 모델 안에 넣는다(multi-token prediction). 이러면 계산기가 덜 논다. 다만 문맥이 길어지면 병목이 weight 읽기에서 KV cache 읽기로 옮겨갈 뿐 사라지지는 않는다.

계산 성능은 지수로 늘었는데 메모리 대역폭은 그만큼 못 늘었다. 칩 설계 문제가 "숫자를 곱셈기까지 얼마나 빨리 가져다주느냐"로 정리되는 이유다. 이게 memory wall이고, 아래 여섯 회사는 이 문제를 각자 다른 방법으로 넘으려 한다.

---

## 3. NVIDIA GPU

2007년에 쓴 CUDA 코드가 지금 Blackwell에서 그대로 컴파일되고 돈다. 18년째 추상화가 거의 안 변했다. 이게 NVIDIA의 답이다. 프로그래밍 방식은 건드리지 않고 세대마다 가속 기능만 얹는다. 같은 칩이 학습도 하고 추론도 하고 그래픽도 하고 과학 계산도 한다.

| 연도 | 아키텍처 | 핵심 |
| --- | --- | --- |
| 2006 | Tesla (G80) | 최초의 CUDA GPU |
| 2016 | Pascal (P100) | NVLink, HBM2, 16비트 연산 — 첫 딥러닝용 설계 |
| 2017 | Volta (V100) | 최초의 Tensor Core |
| 2020 | Ampere (A100) | TF32, 희소 연산, MIG |
| 2022 | Hopper (H100) | 8비트 연산, Transformer Engine, HBM3 |
| 2024 | Blackwell (B200) | 4비트 연산, 칩 2개를 하나로 붙임, NVLink 5 |
| 2025 | Blackwell Ultra (B300) | HBM 288 GB — 긴 문맥 추론 대응 |
| 2026 | Rubin | HBM4, Vera CPU, prefill 전용 칩(Rubin CPX) 분리 |
| 2027 | Rubin Ultra | 칩 4개 패키지, 랙 하나가 600 kW |

### 명령 내리기가 점점 가벼워졌다

계산 코어(SM) 개수는 V100 80개에서 Rubin 224개까지 늘었다. 그런데 코어 하나의 구조는 거의 그대로다. 정작 바뀐 건 행렬 곱셈 명령을 누가 내리느냐였다.

| 세대 | 명령을 내리는 주체 | 동작 |
| --- | --- | --- |
| Volta | 스레드 32개가 같이 | 끝날 때까지 기다린다 |
| Hopper | 스레드 128개가 같이 | 명령만 던지고 바로 다른 일 |
| Blackwell | 스레드 하나가 | 주소표만 넘기고 바로 다른 일 |

곱셈이 뒤에서 도는 동안 나머지 스레드가 softmax를 돌리거나 다음 데이터를 미리 가져올 수 있다는 뜻이다. FlashAttention이 빨라지는 원리가 정확히 이거다. 명령은 점점 커지는데 명령을 내리는 쪽은 할 일이 줄었다.

데이터를 옮기는 일도 계속 스레드에서 떼어냈다. Hopper부터는 전담 엔진(TMA)이 있어서, 스레드 하나가 "이런 모양으로 가져와" 하고 주소표만 던지면 주소 계산은 엔진이 한다. 세대마다 스레드가 할 일이 하나씩 줄어드는 흐름이다.

메모리는 층이 많다. HBM은 V100 32 GB, H100 80 GB, B200 192 GB, B300 288 GB로 늘었고, L2 캐시는 6 MB에서 60 MB로, 코어 안에는 256 KB짜리 공간이 따로 있어서 커널을 띄울 때 캐시로 쓸지 메모장으로 쓸지 정한다. Blackwell부터는 곱셈 결과만 쌓아두는 전용 공간(TMEM)도 생겼다.

숫자 표현은 32비트에서 16, 8, 4비트로 세대마다 절반씩 잘렸다. 그냥 자르면 정확도가 깨지니까 숫자를 작은 묶음으로 나눠 묶음마다 배율을 따로 붙여서 되산다. 비트를 반으로 자를 때마다 같은 전력으로 대략 2배를 번다.

### 연결 — 여기부터가 본론이다

두 가지를 구분해야 한다. scale-up은 GPU 여러 개를 묶어 메모리를 공유하게 만드는 것이다. NVLink로 옆 GPU 메모리를 내 것처럼 읽고 쓰고, 지연은 나노초 단위다. scale-out은 그 묶음들을 랙 밖으로 네트워킹하는 것이다. 주소 공간이 갈라지고 명시적으로 데이터를 보내야 하며 지연은 마이크로초로 뛴다.

여기서 규칙이 하나 나오는데, 여섯 회사 전부 똑같다. 통신량이 많은 작업(tensor 병렬, MoE 라우팅)은 scale-up 안에 가두고, 통신이 적은 작업(data 병렬, pipeline 병렬)만 scale-out으로 내보낸다.

<figure style="margin:36px 0">
<svg viewBox="0 0 740 312" role="img" aria-label="랙 안은 NVLink와 구리로 묶고 랙 사이는 광케이블로 잇는다. 통신 많은 병렬화는 랙 안에 가둔다" style="width:100%;height:auto;font-family:var(--sans,system-ui,sans-serif)">
<defs>
<marker id="n-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
</marker>
</defs>
<rect x="20" y="46" width="270" height="180" rx="10" fill="currentColor" fill-opacity="0.04" stroke="currentColor" stroke-opacity="0.3"/>
<text x="155" y="36" font-size="13" font-weight="600" text-anchor="middle" fill="currentColor">랙 1 (scale-up)</text>
<g>
<rect x="40" y="66" width="52" height="40" rx="6" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="102" y="66" width="52" height="40" rx="6" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="164" y="66" width="52" height="40" rx="6" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="226" y="66" width="44" height="40" rx="6" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.3"/>
</g>
<g font-size="11" text-anchor="middle" fill="currentColor" fill-opacity="0.75">
<text x="66" y="91">GPU</text><text x="128" y="91">GPU</text><text x="190" y="91">GPU</text><text x="248" y="91">…</text>
</g>
<g stroke="var(--accent,#4f46e5)" stroke-width="2.5">
<line x1="66" y1="112" x2="66" y2="140"/><line x1="128" y1="112" x2="128" y2="140"/>
<line x1="190" y1="112" x2="190" y2="140"/><line x1="248" y1="112" x2="248" y2="140"/>
</g>
<rect x="40" y="140" width="230" height="30" rx="7" fill="var(--accent-soft,rgba(79,70,229,0.1))" stroke="var(--accent,#4f46e5)" stroke-opacity="0.5"/>
<text x="155" y="160" font-size="12" text-anchor="middle" fill="var(--accent,#4f46e5)">NVLink 스위치 · 구리</text>
<text x="155" y="192" font-size="11.5" text-anchor="middle" fill="currentColor" fill-opacity="0.7">지연 나노초 · 메모리를 서로 읽는다</text>
<text x="155" y="211" font-size="11.5" text-anchor="middle" fill="currentColor" fill-opacity="0.7">tensor 병렬, MoE 라우팅은 여기 안에</text>
<rect x="450" y="46" width="270" height="180" rx="10" fill="currentColor" fill-opacity="0.04" stroke="currentColor" stroke-opacity="0.3"/>
<text x="585" y="36" font-size="13" font-weight="600" text-anchor="middle" fill="currentColor">랙 2 (scale-up)</text>
<g>
<rect x="470" y="66" width="52" height="40" rx="6" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="532" y="66" width="52" height="40" rx="6" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="594" y="66" width="52" height="40" rx="6" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="656" y="66" width="44" height="40" rx="6" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.3"/>
</g>
<g font-size="11" text-anchor="middle" fill="currentColor" fill-opacity="0.75">
<text x="496" y="91">GPU</text><text x="558" y="91">GPU</text><text x="620" y="91">GPU</text><text x="678" y="91">…</text>
</g>
<g stroke="var(--accent,#4f46e5)" stroke-width="2.5">
<line x1="496" y1="112" x2="496" y2="140"/><line x1="558" y1="112" x2="558" y2="140"/>
<line x1="620" y1="112" x2="620" y2="140"/><line x1="678" y1="112" x2="678" y2="140"/>
</g>
<rect x="470" y="140" width="230" height="30" rx="7" fill="var(--accent-soft,rgba(79,70,229,0.1))" stroke="var(--accent,#4f46e5)" stroke-opacity="0.5"/>
<text x="585" y="160" font-size="12" text-anchor="middle" fill="var(--accent,#4f46e5)">NVLink 스위치 · 구리</text>
<rect x="316" y="118" width="108" height="54" rx="8" fill="currentColor" fill-opacity="0.07" stroke="currentColor" stroke-opacity="0.4"/>
<text x="370" y="140" font-size="11.5" text-anchor="middle" fill="currentColor">IB / Ethernet</text>
<text x="370" y="157" font-size="11.5" text-anchor="middle" fill="currentColor">스위치</text>
<line x1="272" y1="145" x2="312" y2="145" stroke="currentColor" stroke-width="2" stroke-dasharray="6 3" marker-end="url(#n-ar)"/>
<line x1="468" y1="145" x2="428" y2="145" stroke="currentColor" stroke-width="2" stroke-dasharray="6 3" marker-end="url(#n-ar)"/>
<text x="370" y="100" font-size="11.5" text-anchor="middle" fill="currentColor" fill-opacity="0.75">광케이블</text>
<text x="370" y="196" font-size="11.5" text-anchor="middle" fill="currentColor" fill-opacity="0.7">지연 마이크로초</text>
<text x="370" y="213" font-size="11.5" text-anchor="middle" fill="currentColor" fill-opacity="0.7">data · pipeline</text>
<text x="370" y="230" font-size="11.5" text-anchor="middle" fill="currentColor" fill-opacity="0.7">병렬만 여기로</text>
<line x1="300" y1="252" x2="440" y2="252" stroke="currentColor" stroke-opacity="0.35" stroke-dasharray="4 4"/>
<text x="370" y="272" font-size="12" text-anchor="middle" fill="currentColor" fill-opacity="0.8">구리는 여기까지 — 200G/레인에서 약 2 m</text>
<text x="370" y="296" font-size="11.5" text-anchor="middle" fill="currentColor" fill-opacity="0.5">그래서 랙 모양을 토폴로지가 아니라 케이블 길이가 정한다</text>
</svg>
<figcaption style="font-size:13px;color:var(--text-muted);margin-top:12px;line-height:1.65">랙 안은 구리로 묶고 랙 사이는 광으로 잇는다. 이름만 NVLink · ICI · UALink · NeuronLink로 다를 뿐 여섯 회사가 전부 이 구조다.</figcaption>
</figure>

NVIDIA의 scale-up 단위는 GB200 NVL72다. GPU 72개와 Grace CPU 36개, HBM 13.5 TB가 하나의 주소 공간으로 묶인다. 2026년 NVL144, 2027년 NVL576으로 간다.

랙 하나를 GPU 한 장처럼 만들어주는 건 구리 케이블이다. NVL72는 케이블 5,184가닥(랙당 약 2마일)으로 GPU 72개를 초당 130 TB로 잇는다. 광으로 바꾸면 랙당 20 kW를 더 쓴다. 대신 구리는 2 m를 못 넘는다.

이 제약이 만든 결과가 재밌다. NVIDIA가 NVL576을 위해 Kyber라는 새 섀시를 설계했는데, 높이를 기존의 두 배로 잡은 이유가 "GPU 576개를 전부 구리가 닿는 거리 안에 욱여넣기 위해서"다. 토폴로지가 아니라 물리적 케이블 길이가 랙 모양을 정한 셈이다.

랙을 넘는 구간은 광인데, 모듈 하나하나에 레이저가 들어 있어서 수천 GPU짜리 클러스터면 레이저 전력만 수십 kW다. Rubin 세대에서는 이 광 부품을 스위치 칩 안으로 집어넣는다. NVIDIA 주장으로 레이저 수는 1/4, 링크 전력은 1/3.5로 준다.

### 해자는 CUDA 문법이 아니다

CUDA가 18년째 안 변한 건 장점이자 족쇄다. 너무 많은 코드가 얹혀 있어서 NVIDIA도 코어 구조를 확 갈아엎지 못한다.

해자는 다른 데 있다고 본다. 하나는 이 스택 대부분을 NVIDIA가 월급 주지 않는 사람들이 쓴다는 것이다. 20년치 남이 만든 라이브러리와 그걸 배운 개발자 수백만 명이다. 다른 하나는 칩만 파는 게 아니라 사람을 같이 보낸다는 점이다. 자사 엔지니어 수십 명이 주요 고객사 안에 상주하면서 새 모델이 나올 때마다 커널을 쓰고 새 칩이 나올 때마다 튜닝한다. NVIDIA를 떠난다는 건 코드를 다시 쓰는 문제가 아니라 지금 우리 사무실에 앉아 있는 그 엔지니어를 잃는 문제다.

물론 대가도 있다. 스케줄러와 캐시에 들어간 트랜지스터는 곱셈에 못 쓴 트랜지스터다. NVIDIA는 그 세금을 인정하고 대신 곱셈 유닛을 키워서 비율을 맞추는 쪽을 택했다. 다음 장에 나오는 회사는 반대로 간다.

## 4. Google TPU

TPU에는 캐시가 없다. 오타가 아니다. 하드웨어 스케줄러도 없고 스레드 개념도 없다. 아무거나 돌리는 칩 대신 딱 하나, 큰 격자 위의 행렬 곱셈만 잘하고 나머지는 컴파일러가 사이클 단위로 미리 짠다. 그래픽도 안 하고 과학 계산도 안 한다. Google 자기 워크로드를 전기 덜 쓰고 돌리려고 만든 칩이다.

| 연도 | 세대 | 핵심 |
| --- | --- | --- |
| 2015 | v1 | 최초의 양산 딥러닝 전용 칩, 추론만 |
| 2017 | v2 | 학습 지원 시작 |
| 2020 | v4 | 광 스위치(Palomar) 도입, 4,096칩 pod |
| 2023 | v5e / v5p | 효율형과 성능형으로 갈라짐 |
| 2024 | Trillium (v6e) | 곱셈 격자를 256×256으로, Gemini 2.0 학습 |
| 2025 | Ironwood (v7) | 추론용, 9,216칩 묶음 |
| 2026 | v8t / v8i | 학습형/추론형, 9,600칩 묶음 |

### 데이터가 지나가면서 곱해진다

핵심은 MXU라는 격자다. 크기는 128×128 또는 256×256인데 동작 방식이 특이하다. weight를 격자 칸마다 하나씩 미리 깔아두고, 데이터를 왼쪽에서 흘려보낸다. 데이터가 한 칸씩 이동하면서 그 칸에 깔린 weight와 곱해지고 결과는 아래로 흘러내려 쌓인다.

<figure style="margin:36px 0">
<svg viewBox="0 0 740 326" role="img" aria-label="시스톨릭 배열: weight를 칸마다 미리 깔아두고 데이터가 가로로 흐르며 곱해지고 부분합이 아래로 쌓인다" style="width:100%;height:auto;font-family:var(--sans,system-ui,sans-serif)">
<defs>
<marker id="s-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
</marker>
<marker id="s-ar-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--accent,#4f46e5)"/>
</marker>
</defs>
<text x="250" y="34" font-size="12" text-anchor="middle" fill="currentColor" fill-opacity="0.65">weight를 칸마다 하나씩 미리 깔아둔다</text>
<g>
<rect x="190" y="50" width="56" height="56" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="254" y="50" width="56" height="56" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="318" y="50" width="56" height="56" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="382" y="50" width="56" height="56" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="190" y="114" width="56" height="56" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="254" y="114" width="56" height="56" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="318" y="114" width="56" height="56" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="382" y="114" width="56" height="56" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="190" y="178" width="56" height="56" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="254" y="178" width="56" height="56" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="318" y="178" width="56" height="56" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="382" y="178" width="56" height="56" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-opacity="0.3"/>
</g>
<g font-size="13" text-anchor="middle" fill="currentColor" fill-opacity="0.8">
<text x="218" y="83">w</text><text x="282" y="83">w</text><text x="346" y="83">w</text><text x="410" y="83">w</text>
<text x="218" y="147">w</text><text x="282" y="147">w</text><text x="346" y="147">w</text><text x="410" y="147">w</text>
<text x="218" y="211">w</text><text x="282" y="211">w</text><text x="346" y="211">w</text><text x="410" y="211">w</text>
</g>
<g stroke="var(--accent,#4f46e5)" stroke-width="2">
<line x1="104" y1="78" x2="184" y2="78" marker-end="url(#s-ar-a)"/>
<line x1="104" y1="142" x2="184" y2="142" marker-end="url(#s-ar-a)"/>
<line x1="104" y1="206" x2="184" y2="206" marker-end="url(#s-ar-a)"/>
</g>
<g stroke="var(--accent,#4f46e5)" stroke-width="1.5" stroke-opacity="0.55">
<line x1="248" y1="78" x2="252" y2="78"/><line x1="312" y1="78" x2="316" y2="78"/><line x1="376" y1="78" x2="380" y2="78"/>
</g>
<text x="52" y="74" font-size="12" text-anchor="middle" fill="var(--accent,#4f46e5)">데이터</text>
<text x="52" y="90" font-size="11" text-anchor="middle" fill="var(--accent,#4f46e5)" fill-opacity="0.75">한 줄씩</text>
<text x="140" y="228" font-size="11" text-anchor="middle" fill="var(--accent,#4f46e5)" fill-opacity="0.8">1 사이클에</text>
<text x="140" y="243" font-size="11" text-anchor="middle" fill="var(--accent,#4f46e5)" fill-opacity="0.8">한 칸씩 이동</text>
<g stroke="currentColor" stroke-width="1.6" stroke-opacity="0.5">
<line x1="218" y1="240" x2="218" y2="268" marker-end="url(#s-ar)"/>
<line x1="282" y1="240" x2="282" y2="268" marker-end="url(#s-ar)"/>
<line x1="346" y1="240" x2="346" y2="268" marker-end="url(#s-ar)"/>
<line x1="410" y1="240" x2="410" y2="268" marker-end="url(#s-ar)"/>
</g>
<rect x="190" y="274" width="248" height="34" rx="8" fill="currentColor" fill-opacity="0.05" stroke="currentColor" stroke-opacity="0.25"/>
<text x="314" y="296" font-size="12" text-anchor="middle" fill="currentColor" fill-opacity="0.75">부분합이 아래로 쌓인다</text>
<line x1="470" y1="96" x2="470" y2="228" stroke="currentColor" stroke-opacity="0.15"/>
<text x="492" y="112" font-size="12.5" font-weight="600" fill="currentColor">격자 안에서는</text>
<text x="492" y="131" font-size="12.5" font-weight="600" fill="currentColor">메모리를 안 건드린다</text>
<text x="492" y="158" font-size="11.5" fill="currentColor" fill-opacity="0.6">weight는 지나가는 모든</text>
<text x="492" y="175" font-size="11.5" fill="currentColor" fill-opacity="0.6">데이터에 재사용되고,</text>
<text x="492" y="192" font-size="11.5" fill="currentColor" fill-opacity="0.6">데이터는 한 줄을 지나며</text>
<text x="492" y="209" font-size="11.5" fill="currentColor" fill-opacity="0.6">칸 수만큼 재사용된다.</text>
</svg>
<figcaption style="font-size:13px;color:var(--text-muted);margin-top:12px;line-height:1.65">TPU가 메모리 접근을 줄이는 방식. 실제 격자는 128×128 또는 256×256인데 그림은 4×3으로 줄였다.</figcaption>
</figure>

이 구조의 핵심은 격자에 들어간 다음부터 메모리를 한 번도 안 건드린다는 데 있다. weight는 지나가는 모든 데이터에 재사용되고, 데이터는 한 줄을 통과하면서 칸 수만큼 재사용된다. 재사용을 캐시가 판단해주는 게 아니라 배선에 박아둔 셈이다.

왜 그렇게까지 하냐면, 비싼 건 곱셈이 아니라 메모리 접근이기 때문이다. 메모리를 한 번 읽는 데 곱셈 한 번보다 100~1000배 전력이 든다.

대가는 있다. 256×256 격자에 128×128짜리 일을 시키면 칩의 75%가 논다. 그래서 컴파일러가 계산을 무조건 128(또는 256)의 배수로 맞춰 잘라 넣고, 모델 짜는 사람도 그 숫자를 의식하면서 짠다.

### 캐시가 없으면 뭐가 있나

HBM은 v5p 95 GB, Ironwood 192 GB, v8 세대 216~288 GB다. 그 아래에 VMEM이라는 메모장이 있는데 곱셈기한테 데이터를 먹이는 용도다. v4에서 32 MiB였던 게 v8i에서 384 MiB가 됐다. KV cache를 통째로 칩 안에 두겠다는 뜻이다.

모든 데이터는 컴파일 시점에 "너는 여기 산다"가 정해진다. 컴파일러가 "쓰이기 직전에 도착하도록" 전송을 미리 깔아두고, 하드웨어는 미리 가져오지도 내보내지도 않는다. 컴파일러가 맞히면 격자가 안 멈추고, 틀리면 대신 막아줄 게 없다.

추천 모델은 얘기가 다르다. 거대한 표에서 수십억 개 항목을 찾아오는 작업이라 접근이 불규칙하고, 곱셈 격자에는 정확히 안 맞는 모양이다. Google의 답은 SparseCore라는 별도 블록을 옆에 붙이는 거였다. 칩 면적과 전력의 5%만 쓰는데 추천 모델에서 5~7배가 빨라진다. 이 발상은 뒤에 다른 회사에서도 그대로 나온다.

### 연결 — NVIDIA와 정반대

NVLink가 "옆 GPU 메모리를 내 메모리처럼" 보이게 만든다면, Google의 ICI는 그냥 메시지를 주고받는다. 옆 칩 메모리를 직접 읽는 기능 자체가 없다. 칩 사이 통신은 전부 컴파일러가 만들어 넣은 명시적 전송이다.

모양도 다르다. 스위치로 모으는 대신 이웃끼리 직접 잇는 격자(torus)를 만든다. 칩 64개가 한 묶음이고 묶음 안은 구리, 묶음 사이는 광이다.

여기서 NVIDIA에 대응물이 없는 부품이 나온다. Palomar라는 광 스위치인데, 작은 거울이 물리적으로 돌아가면서 광섬유 연결을 바꾼다. 바꾸는 데 밀리초가 걸리는데 상관없다. 작업 시작할 때 모양을 정해놓고 일주일 돌린 다음 다음 작업에 맞춰 다시 바꾸는 식이기 때문이다.

이 부품 하나가 세 가지를 동시에 푼다. 작업마다 연결 모양을 바꿀 수 있고, 큰 묶음을 잘라 나눠 쓸 수 있고, 칩이 죽으면 예비 묶음을 광학적으로 끼워 넣어 작업을 안 죽이고 이어갈 수 있다. 네트워크 하는 사람 눈에는 마지막 게 제일 부럽다.

scale-up 단위는 superpod이다. 역할은 NVL72와 같은데 규모가 두 자릿수 크다. Ironwood 기준 9,216칩이 한 묶음이다.

v8i는 예외다. 1,024칩 규모에서 torus를 버리고 계층형으로 갈아탔는데, 이유가 교과서적이다. torus는 이웃끼리 주고받는 통신에 최강인데 MoE는 정반대로 전부가 전부에게 보낸다. 이럴 땐 가장 먼 두 칩 사이 거리가 전체 속도를 정한다. 16홉이 7홉으로 줄었다. (그림은 11장에 있다.)

### 컴파일러가 유일한 인터페이스다

GPU에서는 개발자가 커널을 쓰고 컴파일러는 거든다. TPU에서는 개발자가 JAX로 수식만 쓰고 그 아래 전부를 XLA가 책임진다. 뭘 합칠지, 데이터가 어디 살지, 언제 옮길지, 수천 칩에 어떻게 나눌지까지 전부다.

탈출구가 없진 않다. Pallas라는 커널 작성 언어가 있어서 컴파일러가 최적해를 못 찾을 때 직접 쓸 수 있다.

CUDA와 비교하면 TPU 생태계는 흩어져 있지 않고 모여 있다. XLA, JAX, Flax, Pallas 전부 Google이 직접 만들어 오픈소스로 낸다. 남이 만든 코드는 훨씬 적다. 내 워크로드가 Gemini를 닮을수록 유리하고 이상할수록 불리하다는 뜻이기도 하다.

## 5. AMD Instinct

AMD는 반대로 갔다. 코어를 2012년 이후 거의 그대로 두고 그 돈을 메모리와 패키징에 썼다. 2021년부터 매 세대 HBM 용량이 같은 시기 NVIDIA 제품 이상이고, 데이터센터 GPU 중 처음으로 칩을 3D로 쌓았고, CPU와 GPU가 메모리를 진짜로 공유하는 제품도 먼저 냈다.

용어가 달라서 처음엔 헷갈렸다.

| AMD | NVIDIA |
| --- | --- |
| Compute Unit (CU) | Streaming Multiprocessor (SM) |
| Wavefront (64개) | Warp (32개) |
| Matrix Core | Tensor Core |
| LDS | Shared Memory |
| Infinity Fabric | NVLink |

### 곱셈 명령이 제자리에 머물렀다

3장에서 NVIDIA의 곱셈 명령이 "스레드 32개 → 128개 → 한 개"로 가벼워졌다고 했다. AMD는 이 길을 안 갔다. 2020년부터 2025년까지 모든 세대가 스레드 64개가 같이 명령을 내리는 방식이다.

대가가 둘이다. 하나는 낭비다. 64개짜리 묶음을 반만 채우면 32개가 논다. NVIDIA는 32개 묶음이라 16개만 논다. 다른 하나가 더 아프다. NVIDIA는 곱셈을 던져놓고 다른 일을 할 수 있는데 AMD는 그게 안 된다. 곱셈을 시킨 그 묶음은 끝날 때까지 의미 있는 다른 일을 못 한다.

얼마나 손해인지는 상황에 달렸다. 학습처럼 곱셈만 계속하는 구간에서는 어차피 다른 할 일이 없어서 차이가 작다. AMD가 슈퍼컴퓨터 시장에서 앞섰던 이유이기도 하다(Frontier, El Capitan). 반대로 attention은 곱셈과 softmax가 계속 번갈아 나오고 그 겹침이 커널 구조 그 자체다. 여기서 AMD는 파이프라인을 손으로 다시 만들어야 한다.

### 메모리로는 이기고 있다

Infinity Cache가 NVIDIA에 없는 물건이다. MI300X 기준 256 MB인데 속도가 초당 12 TB로 같은 칩 HBM(5.3 TB/s)보다 두 배 이상 빠르다. 원래 게이밍 GPU에서 좁은 메모리 버스를 보완하려고 만든 기술인데 LLM의 재사용 패턴에 잘 맞아서 그대로 가져왔다. NVIDIA는 HBM 대역폭을 키우는 쪽에, AMD는 캐시에 걸었다.

HBM 용량도 공격적이다. MI300X 192 GB, MI325X 256 GB, MI350X 288 GB. 실제로 이게 통한 구간이 있다. MI300X 8장이면 HBM이 1.5 TB라서 405B 모델을 서버 한 대에 통째로 올린다. 같은 모델을 H100 8장(640 GB)에 올리려면 쪼개는 작업이 필요하다.

### 랙 단위 답이 2026년까지 없었다

문제는 학습이다. 학습은 랙 전체를 하나처럼 써야 하는데 AMD는 MI355X까지도 8장짜리 서버가 전부였다. NVIDIA가 72장을 묶는 동안 8장에 머물렀다.

Helios가 그 공백을 메우러 온다. 2026년 하반기, 랙당 GPU 72개, HBM 31 TB 규모다. 연결에는 UALink라는 개방 표준을 쓴다. AMD가 Apple, AWS, Cisco, Google, Intel, Meta, Microsoft와 같이 만든 규격이고 약속은 "NVLink급인데 누구의 소유도 아닌 것"이다.

그런데 UALink용 스위치 칩이 2027년까지 양산이 안 된다. 그래서 Helios는 출시 시점에 기존 연결 방식을 Ethernet에 터널링해서 흘려보낸다. 출시 때의 Helios는 NVL72 같은 진짜 공유 메모리 묶음이라기보다 빠른 Ethernet 터널에 가깝다는 뜻이다. 2026년에 제품을 내놓기 위해 치른 타협이다.

랙 바깥은 전부 Ethernet이다. AMD는 InfiniBand를 아예 안 판다. UEC라는 개방 표준에 정박해 있는데 여기서 나온 UET이 핵심이다. RoCEv2가 InfiniBand 방식을 Ethernet 봉투에 담은 것이라면, UET은 AI 트래픽을 위해 처음부터 다시 설계했다. 패킷을 여러 경로에 뿌리고, 빠진 것만 골라 재전송하고, 혼잡 제어도 현대적이다.

다만 스위치 칩은 AMD 것이 아니다. Helios의 scale-out은 Broadcom Tomahawk 6를 통과한다. NVIDIA가 스위치부터 NIC, 광 부품까지 다 자기 것으로 채우는 것과 대조적이다. AMD는 NIC 한 층만 갖고 나머지는 표준과 파트너에 건다.

업계는 AMD 쪽으로 움직였다. 2025년 AI 네트워크 물량에서 Ethernet이 InfiniBand의 두 배가 넘는다. AWS, Microsoft, Meta, Oracle, xAI 전부 Ethernet으로 표준화했다. 남은 질문은 Ethernet이 IB를 따라잡느냐가 아니라 Helios가 랙 단위 격차를 얼마나 빨리 좁히느냐다.

### ROCm은 어디까지 왔나

NVIDIA 스택이 자기들만 관리하는 폐쇄형이라면 ROCm은 GitHub에 다 열려 있다. CUDA 코드를 자동 변환해주는 도구도 있어서 과학 계산 코드는 80~95%가 그대로 넘어간다.

문제는 최신 AI 커널이다. NVIDIA 최신 칩에만 있는 기능을 파고드는 코드는 대응물이 없어 손으로 다시 써야 한다. FlashAttention이 그 예다. FA2는 잘 돌고, FA3는 부분 지원이고, FA4(2026년 3월)는 포팅 자체가 없다.

숫자로 보면 2026년 3월 독립 벤치마크에서 ROCm이 같은 조건 CUDA보다 10~25% 느리다. 기능은 따라잡았는데 성능은 아직이라는 뜻이다. 그래도 파일럿이 아니라 실제 서빙 fleet에 들어가 있다. Azure에서 OpenAI가 추론을 돌리고 Meta가 Llama 추론을 돌린다.

## 6. Cerebras

Cerebras의 진단이 여섯 중에 제일 인상적이었다. memory wall은 웨이퍼를 잘라서 생긴 문제라는 것이다.

생각해보면 그렇다. 공장은 둥근 실리콘 판에 칩 수십 개를 찍고 톱으로 잘라낸다. 그리고 업계는 그 조각들을 다시 붙이려고 온갖 기술을 쏟아붓는다. HBM, NVLink, 랙당 구리 5,184가닥. 그렇게 붙여봐야 원래 칩 안에서의 속도에는 한참 못 미친다. Cerebras는 그 톱질을 건너뛴다.

칩 하나가 말 그대로 웨이퍼 한 장이다. 면적 46,225 mm², 작은 코어 90만 개, 메모리는 전부 SRAM 44 GB다.

1980년대에 이 방식이 실패한 이유는 수율이었다. 웨이퍼 하나에 결함이 하나만 있어도 전체가 죽는다. Cerebras의 답은 잘게 쪼개기였다. H100이라면 결함 하나가 6 mm²짜리 코어를 통째로 죽이는데 여기서는 0.05 mm²짜리 하나만 죽는다. 애초에 97만 개를 만들어 90만 개만 출하한다. 7%는 예비고, 결함 난 자리는 하드웨어가 알아서 우회해 온전한 격자를 복원한다.

### 곱셈 전용 유닛이 없다

NVIDIA, Google, AMD는 전부 곱셈을 전담하는 큰 유닛에 성능을 몰아준다. Cerebras에는 그게 없다. 코어 90만 개가 나눠서 한다. weight가 도착하면 코어들에 뿌려지고 각 코어가 자기 몫을 곱한 다음 결과를 격자를 가로질러 모은다.

공짜로 얻는 것도 하나 있다. 0을 건너뛰는 것이다. 이 구조는 데이터가 도착해야 계산이 시작되는데, 0은 보내는 쪽에서 걸러버리면 받는 쪽이 아예 보지도 않는다. 다른 칩들이 특수 모드로 겨우 지원하는 기능이 여기서는 기본이다. 다만 아직 안 써먹은 카드다. 이걸 활용한 대표 모델이 아직 없고, 가장 큰 고객 모델도 이 기능을 안 쓴다.

데이터 흐름도 반대다. GPU와 TPU에서는 weight가 칩에 머물고 데이터가 흘러가는데, Cerebras에서는 데이터가 머물고 weight가 흘러간다. 원본 weight는 옆에 붙은 별도 장비(MemoryX)에 두고 레이어 단위로 웨이퍼를 가로질러 흘려보낸다. 덕분에 모델 크기가 44 GB에 묶이지 않고, 70B 모델도 쪼개기 없이 장비 하나짜리 프로그램으로 짤 수 있다.

### 메모리 계층이 하나다

코어마다 48 kB씩, 합쳐서 44 GB. 그게 전부다. HBM 없고 L2 없고 데이터를 내보내는 규칙도 없다.

자주 인용되는 "초당 21 PB"는 걸러 봐야 한다. 코어 90만 개의 메모리 포트를 전부 더한 값이라 HBM 수치와 나란히 놓고 비교할 게 아니다. 정직한 비교는 "계산 한 번에 몇 바이트를 먹일 수 있나"다. 이 기준으로 웨이퍼는 1.3 바이트, B200은 0.002 바이트다. 이 축에서는 모든 GPU가 굶고 있고 웨이퍼만 균형이 맞는다. 토큰 하나 뽑을 때마다 모델 전체를 읽어야 하는 decode야말로 웨이퍼가 맞춰 태어난 상황이다.

문제는 바깥이다. 웨이퍼가 외부와 연결되는 통로가 초당 1.2 Tb밖에 안 된다. Blackwell GPU 한 장에 붙는 NIC 하나보다 조금 많은 수준이다. 칩 안과 칩 밖 사이에 10만 배 차이가 있다. NVIDIA는 계층이 완만하게 내려가는데 여기는 절벽이다.

게다가 섬이 자라지 않는다. 최신 공정에서 SRAM 밀도는 사실상 안 줄어든다. WSE-3는 공정을 한 단계 줄이고 트랜지스터를 54% 늘렸는데 SRAM은 10%밖에 안 늘었다. 이 구조에서 제일 부족한 자원이, 공정 발전이 더 이상 사주지 않는 바로 그것이다.

### 속도는 진짜인데 비용이 문제다

실측이 있다. 2024년 출시 시점에 Llama 3.1 8B가 초당 1,850 토큰, 70B가 446 토큰이었다. 2025년 Llama 4 Maverick은 2,522 토큰으로 같은 시기 Blackwell 최고 기록의 2.4배다. 사용자 한 명 기준 속도에서 근접하는 GPU 사업자가 없다.

대신 경제성이 날카롭다. 웨이퍼 한 장에 44 GB라 큰 모델은 장비를 여러 대 먹는다. 1.6T급 모델이면 24대가 필요한데 GPU로는 랙 몇 개면 되는 모델이다. 같은 오픈 모델 기준 토큰당 가격이 GPU 사업자의 3~5배다. 문맥 길이도 131K에서 막히는데 경쟁사들은 256K~1M을 서빙한다. Llama 405B는 조용히 API에서 내려갔다.

그럼에도 시장이 값을 치르고 있다. 레이턴시가 중요한 곳들이 쓰고, 2026년 1월에 OpenAI가 2028년까지 750 MW 규모로 계약했다. 웨이퍼 방식이 받은 최대 규모 보증이다.

소프트웨어는 여섯 중에 제일 약하다. 컴파일러가 범용 코드 생성기가 아니라 패턴 맞추기다. 미리 만들어둔 커널 목록과 대조해 맞으면 쓰고 안 맞으면 느린 자동 생성 버전으로 떨어진다. 크기가 변하는 텐서 안 되고, 데이터에 따라 갈라지는 분기 안 되고, 중간에 값을 꺼내볼 수도 없다.

무엇보다 직접 커널을 쓸 방법이 없다. 새로운 attention 변종이 나왔을 때 CUDA는 "커널을 써라", TPU는 Pallas, AMD는 Triton이라는 답이 있는데 여기는 "Cerebras 엔지니어를 부르세요"가 답이다. 실제로 이 플랫폼의 대표 모델은 전부 Cerebras 직원이 상주해서 같이 만들었다.

묘한 면역도 있다. FlashAttention은 메모리 계층을 넘나들며 잘게 쪼개는 기법인데 여기는 쪼갤 계층이 없다. AMD를 수년간 괴롭힌 종류의 최적화가 여기서는 아예 해당이 없다. 면역과 빈곤이 같은 사실인 셈이다.

## 7. AWS Trainium

Nitro 카드와 Graviton CPU를 만든 Annapurna Labs가 만들었다. 설계 철학이 솔직하게 "빠른 2등"이다.

계산 코어는 TPU의 검증된 방식을 그대로 가져왔다. 128×128 곱셈 격자, 컴파일러가 관리하는 메모장, 전체 프로그램 사전 컴파일. Google의 컴파일러(XLA)까지 같이 쓴다. 네트워크는 이미 AWS 나머지를 나르고 있는 걸 그대로 쓴다. 진짜 Amazon 것은 좁고 의도적이다. 통신 전용 실리콘 하나, 그리고 AWS 안에서만 NVIDIA보다 싸면 된다는 가격 구조다.

| 연도 | 제품 | 핵심 |
| --- | --- | --- |
| 2015 | Annapurna Labs 인수 | 약 3.5억 달러 |
| 2019 | Inferentia | 첫 ML 칩, 추론만 |
| 2022 | Trainium1 | 첫 학습 칩, HBM 32 GB |
| 2024 | Trainium2 | HBM 96 GB, 64칩 묶음. Anthropic용 Project Rainier 구동 |
| 2025 | Trainium3 | AWS 첫 3 nm, 144칩 묶음 |

조립 방식은 TPU와 다르다. 칩 하나에 NeuronCore가 8개 들어가고, NeuronCore 하나는 단일 덩어리가 아니라 역할이 다른 엔진 묶음이다. 128×128 곱셈 격자(Tensor Engine), 합치는 계산을 맡는 Vector Engine, 원소 하나씩 처리하는 Scalar Engine, 그리고 위 셋 어디에도 안 맞는 것 전부를 처리하는 GPSIMD Engine이다.

잘 컴파일된 프로그램은 이 넷이 동시에 돈다. 곱셈 격자가 계산하는 동안 Vector Engine이 앞 타일의 softmax를 돌리고 전송 엔진이 다음 타일을 준비한다. NVIDIA와 TPU가 소프트웨어로 만드는 겹침을 여기서는 물리적으로 분리된 엔진으로 표현한 셈이다. 대가는 가장자리에서 나온다. 어느 전문 엔진에도 안 맞는 연산자는 범용 경로로 떨어지고 느리다.

메모리는 세 층이고 전부 컴파일러가 관리한다. AWS 문서가 대놓고 "CPU나 GPU와 달리 캐시가 없고 모든 메모리 이동이 프로그램 안에 명시적으로 적힌다"고 쓴다. HBM은 Trainium2 96 GB, Trainium3 144 GB다.

용량으로는 NVIDIA에 밀린다. 96 GB는 H200·B200 아래고 144 GB도 같은 시기 B200(192 GB)·B300(288 GB) 아래다. AWS가 실제로 당기는 레버는 메모리가 아니라 가격이다. 자기가 만들고 자기가 임대하는 칩이라 단가를 마음대로 정할 수 있다.

숫자도 하나 걸러 들어야 한다. AWS는 "4배 빨라졌다"를 앞세우는데 자사 기술 문서에는 8비트 대비 2배라고 적혀 있다. 4배는 16비트 기준이다. 그리고 Trainium3의 4비트는 속도를 안 준다. 격자에 닿기 전에 8비트로 바꿔서 넣기 때문에 메모리만 아끼고 계산은 그대로다.

### 통신을 하드웨어로 뺐다

여기가 제일 특이한 부분이다. 분산 학습은 시간의 상당 부분을 칩끼리 결과를 합치는 데 쓴다. GPU에서는 이 작업이 계산하는 바로 그 코어 위에서 돌기 때문에 통신과 계산이 같은 실리콘을 놓고 싸운다.

Trainium은 이걸 전용 하드웨어로 빼버렸다. 칩당 CC-Core 20개가 통신 포트에 직결돼 있어서, 곱셈 엔진이 계속 도는 동안 결과 합치기를 따로 처리한다. 통신이 "멈춰서 하는 일"이 아니라 "동시에 하는 일"이 된다. Google이 SparseCore로 한 것과 똑같은 수다.

칩을 묶는 건 NeuronLink다. Trainium2는 64칩, Trainium3는 144칩이 한 묶음이다. 바깥은 새로 만들지 않고 AWS가 이미 쓰고 있는 EFA를 그대로 쓴다. InfiniBand는 없다. 칩이 상품이 아니라 클라우드가 상품이고 칩은 부품이기 때문에 가능한 선택이다.

소프트웨어는 Neuron SDK다. Cerebras와 결정적으로 다른 점이 하나 있는데, NKI라는 커널 작성 경로가 있다는 것이다. 컴파일러가 최적해를 못 만들면 직접 쓸 수 있다.

## 8. Groq LPU

미리 밝혀둘 게 있다. 이 섹션은 원문 해당 부분을 확보하지 못해서 Groq의 공개 발표 자료로 재구성했다. 원문이 확실히 밝힌 건 한 문장뿐이다. Groq는 200억 달러 규모로 NVIDIA에 인수됐다.

발상은 극단적이다. 하드웨어에서 예측 불가능한 요소를 전부 없애면 컴파일러가 모든 사이클을 미리 계획할 수 있다는 것이다. 캐시 없고, 미리 실행해보는 것도 없고, 분기 예측도 없고, 코어끼리 순서를 다투는 것도 없다. 창업자가 Google 초대 TPU 설계자였다는 걸 생각하면 계보가 분명하다. TPU의 "컴파일러가 다 정한다"를 네트워크까지 밀어붙였다.

덕분에 실행 시간을 미리 정확히 알 수 있다. 런타임에 뭘 최적화할 필요가 없고 가끔 튀는 느린 응답이 사실상 사라진다.

곱셈기는 320×320 구조로 칩 안에 MACC 409,600개를 갖고 있다. 메모리는 SRAM만 쓴다. 칩당 230 MB에 초당 80 TB다. Cerebras와 같은 계열의 베팅을 훨씬 작은 칩에서 한 셈이다.

네트워크까지 결정론적으로 만들었다. Dragonfly 구조를 쓰면서 순서가 흔들릴 수 있는 요소를 제거했다. 모든 보드의 모든 칩이 같은 박자로 돈다.

칩당 230 MB라는 숫자가 나머지를 다 정한다. 70B 모델 하나를 서빙하려면 칩 수백 개에 레이어를 쪼개 얹어야 한다. Groq에서는 scale-up과 scale-out 구분이 흐릿한데, 길게 늘어선 파이프라인 자체가 모델 하나이기 때문이다.

보상은 명확하다. HBM을 왕복하지 않으니 사용자 한 명 기준 속도가 GPU를 크게 앞선다. 출시 초기 Llama-2 70B에서 초당 300 토큰, 이후 세대에서 500 토큰을 넘겼다. 응답이 실시간으로 느껴지는 체감을 처음 만든 게 이 회사다.

대가는 Cerebras와 같은 종류다. 모델 하나에 칩이 너무 많이 든다. 2024년부터 칩 판매를 그만두고 GroqCloud로 서비스만 파는 쪽으로 돌아섰다.

## 9. 한눈에 비교

여섯 개를 따로 읽고 나면 자꾸 헷갈려서 표로 만들어뒀다.

### 9-1. 네 가지 질문으로 본 여섯 개

| | 데이터가 사는 곳 | 계산 유닛 | 누가 순서를 정하나 | 한 묶음 크기 |
| --- | --- | --- | --- | --- |
| NVIDIA | HBM + 캐시 + 코어 내부 메모리 | Tensor Core | 하드웨어 + 커널 코드 | GPU 72 → 144 → 576개 |
| Google TPU | HBM + 메모장, 캐시 없음 | 곱셈 격자 | 컴파일러 | 칩 9,216 → 9,600개 |
| AMD | HBM + 256 MB 캐시 | Matrix Core | 하드웨어 | GPU 8개 → Helios 72개 |
| Cerebras | SRAM 44 GB 한 층 | 전용 유닛 없음, 코어 90만 개 | 데이터 도착 순서 | 웨이퍼 1장(고정) |
| AWS Trainium | HBM + 메모장, 캐시 없음 | 곱셈 격자 + 보조 엔진 3개 | 컴파일러 | 칩 64 → 144개 |
| Groq | SRAM 230 MB, DRAM 없음 | 320×320 격자 | 컴파일러(네트워크까지) | 길게 늘어선 파이프라인 |

### 9-2. 한 줄로 줄인 각자의 베팅

| 회사 | 베팅 |
| --- | --- |
| NVIDIA | 워크로드는 계속 바뀐다. 전부 프로그래밍 가능하게 두고, 스케줄러에 쓴 트랜지스터 세금은 곱셈 유닛을 키워 상쇄한다 |
| Google TPU | 워크로드는 예측 가능하다. 캐시와 동적 스케줄러를 지우고 그 면적을 전부 곱셈에 쓴다 |
| AMD | 추론은 용량 싸움이다. 메모리를 더 얹고, 독점 규격 대신 개방 표준(UALink, UEC)에 건다 |
| Cerebras | 자르지 않으면 붙일 일도 없다. 용량을 포기하고 대역폭을 산다. 처리량이 아니라 응답 속도를 판다 |
| AWS Trainium | 클라우드가 상품이고 칩은 부품이다. 설계는 빌리고 아낀 힘은 네트워크와 가격에 쓴다 |
| Groq | 불확실한 걸 다 지우면 컴파일러가 전부 계획할 수 있다. 칩을 팔지 말고 토큰을 판다 |

### 9-3. 직접 커널을 쓸 수 있나

| 플랫폼 | 기본 경로 | 직접 쓰는 길 | 최신 최적화 따라가는 속도 |
| --- | --- | --- | --- |
| NVIDIA | CUDA 커널 직접 | 제한 없음 | 여기서 최초 구현이 나온다 |
| AMD | PyTorch → Triton | Triton, AITER 등 | 수개월 지연 |
| Google TPU | JAX → XLA 자동 | Pallas | 직접 작성 가능 |
| AWS Trainium | PyTorch → Neuron | NKI | 가능하지만 fallback은 느림 |
| Cerebras | 패턴 맞추기 | 사실상 없음 | Cerebras 엔지니어에 의존 |
| Groq | 컴파일러 자동 | 사실상 없음 | 컴파일러 개선 대기 |

### 9-4. 네트워크 스택

| 플랫폼 | 묶는 방식 | 메모리 공유? | 바깥 네트워크 | 광 전략 |
| --- | --- | --- | --- | --- |
| NVIDIA | NVLink + 스위치 | 예 | InfiniBand 또는 Spectrum-X | 랙 안은 구리, 밖은 광, Rubin부터 스위치에 내장 |
| Google TPU | ICI 격자(torus) | 아니오 | Virgo + Jupiter | 광 스위치(OCS)를 랙부터 건물까지 동일하게 |
| AMD | UALink(출시 땐 Ethernet 터널) | 목표는 예 | Ethernet + UEC | 파트너(Broadcom) 실리콘 |
| Cerebras | 웨이퍼 내부 격자 | 해당 없음 | 100 GbE 12개 | - |
| AWS Trainium | NeuronLink | 아니오 | EFA(기존 AWS 네트워크) | - |
| Groq | Dragonfly | 아니오 | 같은 네트워크 | - |

## 10. 여섯 개를 겹쳐보면 보이는 것

**숫자를 계속 반으로 자른다.** 32비트에서 16, 8, 4비트로 세대마다 절반이다. 그냥 자르면 정확도가 깨지니 작은 묶음마다 배율을 따로 붙여 되산다. NVIDIA, AMD, TPU가 쓰는 형식이 같은데, 개방 컨소시엄에서 같이 정했기 때문이다. 예외는 16비트를 고수하는 Cerebras, 그리고 4비트가 속도를 안 주는 Trainium3다.

**곱셈 명령을 내리는 일이 점점 가벼워진다.** NVIDIA가 스레드 32개에서 한 개로 걸어왔다는 얘기를 3장에서 했는데, 종착점이 이미 존재했다는 게 흥미롭다. Cerebras 코어에서는 애초에 그 형태밖에 없었다. NVIDIA는 다섯 세대에 걸쳐 그 지점으로 걸어갔고 AMD는 따라오지 않았다. 그 차이가 attention 성능에서 그대로 나온다.

**컴파일러가 다 하느냐, 사람이 커널을 쓰느냐.** TPU, Trainium, Cerebras, Groq는 컴파일러가 전부 정한다. NVIDIA와 AMD는 사람이 커널을 쓴다. 다만 양쪽이 수렴 중이다. GPU 쪽에서는 Triton이 격차를 좁히고, 가속기 쪽에서는 Pallas와 NKI가 커널 경로를 열었다.

**본업이 아닌 일은 옆에 전용 블록을 깎는다.** Google의 SparseCore(추천 모델), AWS의 CC-Core(통신), Cerebras의 0 걸러내기. 셋 다 같은 수다. 메인 코어를 비틀어 맞추지 말고 작은 면적을 떼어 전용 블록을 둔다. GPU에서 통신이 계산하는 코어 위에서 돌며 자리를 다투는 것과 정확히 대비된다.

**묶음 크기 경쟁이 붙었고, 하나만 안 자란다.**

<figure style="margin:36px 0">
<svg viewBox="0 0 740 316" role="img" aria-label="한 묶음에 들어가는 칩 수 변화. NVIDIA 72에서 576, TPU 4096에서 9600, AMD 8에서 72, AWS 64에서 144로 늘었고 Cerebras만 웨이퍼 한 장에 고정이다" style="width:100%;height:auto;font-family:var(--sans,system-ui,sans-serif)">
<defs>
<marker id="g-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
</marker>
</defs>
<g stroke="currentColor" stroke-opacity="0.09">
<line x1="130" y1="40" x2="130" y2="266"/><line x1="275" y1="40" x2="275" y2="266"/>
<line x1="420" y1="40" x2="420" y2="266"/><line x1="565" y1="40" x2="565" y2="266"/>
<line x1="710" y1="40" x2="710" y2="266"/>
</g>
<line x1="130" y1="266" x2="710" y2="266" stroke="currentColor" stroke-opacity="0.25"/>
<g font-size="11" text-anchor="middle" fill="currentColor" fill-opacity="0.5">
<text x="130" y="284">1</text><text x="275" y="284">10</text><text x="420" y="284">100</text>
<text x="565" y="284">1,000</text><text x="710" y="284">10,000</text>
</g>
<text x="420" y="306" font-size="11.5" text-anchor="middle" fill="currentColor" fill-opacity="0.45">한 묶음에 들어가는 칩 수 (로그 눈금)</text>
<g font-size="12.5" text-anchor="end" fill="currentColor">
<text x="118" y="84">NVIDIA</text>
<text x="118" y="124">Google TPU</text>
<text x="118" y="164">AMD</text>
<text x="118" y="204">AWS Trainium</text>
<text x="118" y="244" fill="var(--accent,#4f46e5)" font-weight="600">Cerebras</text>
</g>
<g stroke="currentColor" stroke-width="2" stroke-opacity="0.75">
<line x1="399" y1="80" x2="524" y2="80" marker-end="url(#g-ar)"/>
<line x1="654" y1="120" x2="701" y2="120" marker-end="url(#g-ar)"/>
<line x1="261" y1="160" x2="393" y2="160" marker-end="url(#g-ar)"/>
<line x1="392" y1="200" x2="437" y2="200" marker-end="url(#g-ar)"/>
</g>
<g fill="currentColor" fill-opacity="0.35">
<circle cx="399" cy="80" r="4"/><circle cx="654" cy="120" r="4"/>
<circle cx="261" cy="160" r="4"/><circle cx="392" cy="200" r="4"/>
</g>
<g fill="currentColor">
<circle cx="530" cy="80" r="5"/><circle cx="707" cy="120" r="5"/>
<circle cx="399" cy="160" r="5"/><circle cx="443" cy="200" r="5"/>
</g>
<circle cx="130" cy="240" r="6" fill="var(--accent,#4f46e5)"/>
<g font-size="11" fill="currentColor" fill-opacity="0.5" text-anchor="middle">
<text x="399" y="68">72</text>
<text x="640" y="138">4,096</text>
<text x="261" y="148">8</text>
<text x="386" y="218">64</text>
</g>
<g font-size="12" font-weight="600" fill="currentColor" text-anchor="middle">
<text x="532" y="68">576</text>
<text x="705" y="108" text-anchor="end">9,600</text>
<text x="401" y="148">72</text>
<text x="447" y="188">144</text>
</g>
<text x="146" y="236" font-size="12" font-weight="600" fill="var(--accent,#4f46e5)">웨이퍼 1장</text>
<text x="146" y="252" font-size="11" fill="var(--accent,#4f46e5)" fill-opacity="0.8">2019년 이후 그대로</text>
</svg>
<figcaption style="font-size:13px;color:var(--text-muted);margin-top:12px;line-height:1.65">묶음이 클수록 통신 많은 작업을 느린 네트워크로 안 내보낸다. Cerebras의 1은 칩이 아니라 웨이퍼 한 장이라 단위가 다르지만, 안 자란다는 게 요점이라 같이 놓았다. Groq는 모델마다 달라져서 뺐다.</figcaption>
</figure>

Cerebras만 2019년 이후 웨이퍼 한 장에 고정이다. 300 mm가 업계 최대 웨이퍼라 더 가져올 면적이 없다. 이 크기가 곧 "통신 많은 작업을 느린 네트워크로 안 내보내고 가둘 수 있는 한계"다.

**케이블 길이가 랙 모양을 정한다.** 구리는 2 m를 못 넘는다. 랙 안은 구리, 랙 밖은 광이다. NVIDIA가 NVL576용으로 새 섀시를 설계한 이유가 토폴로지가 아니라 케이블 길이였다는 게 이 제약을 잘 보여준다. 랙 밖 구간은 광 모듈의 레이저 전력만 수십 kW라 Rubin 세대에서 스위치 칩 안으로 접어 넣는다. Google만 완전히 다른 축에 있다. 패킷 스위치가 아니라 거울을 돌리는 광 스위치를 랙부터 건물까지 똑같이 쓴다.

**Ethernet이 이기고 있다.** 2025년 AI 네트워크 물량에서 Ethernet이 InfiniBand의 두 배가 넘는다. 핵심은 UET이다. RoCEv2가 InfiniBand 방식을 Ethernet 봉투에 담은 거라면 UET은 AI용으로 처음부터 다시 설계했다. NVIDIA도 Spectrum-X로 Ethernet 선택지를 준다. 질문은 이제 Ethernet이 따라잡느냐가 아니라 랙 단위 격차를 누가 먼저 닫느냐로 옮겨갔다.

## 11. 네트워크 하는 입장에서 건진 것

여기는 원문에 없는 내 해석이다.

**scale-up이냐 scale-out이냐는 곧 어디에 무엇을 두느냐다.** 여섯 플랫폼이 전부 같은 규칙을 쓴다. 통신 많은 작업은 안에 가두고 적은 작업만 밖으로 보낸다. 이유는 단순하다. 칩당 안쪽 대역이 바깥쪽보다 한두 자릿수 크다. "이 워크로드가 내 네트워크를 얼마나 때릴까"는 모델 구조가 아니라 배치가 정하고, 그 배치는 묶음 크기가 정한다.

**MoE가 토폴로지 기준을 바꾸고 있다.** 4장에서 TPU v8i가 torus를 버렸다고 했는데, 이게 제일 교과서적인 사례다.

<figure style="margin:36px 0">
<svg viewBox="0 0 740 306" role="img" aria-label="격자형 토러스는 먼 칩까지 여러 번 건너가고 계층형 구조는 스위치 두 번이면 닿는다" style="width:100%;height:auto;font-family:var(--sans,system-ui,sans-serif)">
<text x="185" y="28" font-size="13" font-weight="600" text-anchor="middle" fill="currentColor">격자형 (torus)</text>
<text x="555" y="28" font-size="13" font-weight="600" text-anchor="middle" fill="currentColor">계층형 (high-radix)</text>
<g stroke="currentColor" stroke-opacity="0.22" stroke-width="1.5">
<line x1="80" y1="80" x2="290" y2="80"/><line x1="80" y1="140" x2="290" y2="140"/>
<line x1="80" y1="200" x2="290" y2="200"/><line x1="80" y1="260" x2="290" y2="260"/>
<line x1="80" y1="80" x2="80" y2="260"/><line x1="150" y1="80" x2="150" y2="260"/>
<line x1="220" y1="80" x2="220" y2="260"/><line x1="290" y1="80" x2="290" y2="260"/>
</g>
<g stroke="currentColor" stroke-opacity="0.12" stroke-width="1.5" fill="none" stroke-dasharray="3 3">
<path d="M 80 80 C 50 60, 320 60, 290 80"/>
<path d="M 80 260 C 50 288, 320 288, 290 260"/>
<path d="M 80 80 C 56 50, 56 290, 80 260"/>
<path d="M 290 80 C 314 50, 314 290, 290 260"/>
</g>
<polyline points="80,80 150,80 220,80 290,80 290,140 290,200 290,260" fill="none" stroke="var(--accent,#4f46e5)" stroke-width="3" stroke-linejoin="round"/>
<g fill="currentColor" fill-opacity="0.45">
<circle cx="150" cy="140" r="5"/><circle cx="220" cy="140" r="5"/>
<circle cx="80" cy="140" r="5"/><circle cx="80" cy="200" r="5"/>
<circle cx="150" cy="200" r="5"/><circle cx="220" cy="200" r="5"/>
<circle cx="150" cy="260" r="5"/><circle cx="220" cy="260" r="5"/>
</g>
<g fill="var(--accent,#4f46e5)">
<circle cx="150" cy="80" r="5"/><circle cx="220" cy="80" r="5"/>
<circle cx="290" cy="140" r="5"/><circle cx="290" cy="200" r="5"/>
</g>
<circle cx="80" cy="80" r="8" fill="var(--accent,#4f46e5)"/>
<circle cx="290" cy="260" r="8" fill="var(--accent,#4f46e5)"/>
<text x="62" y="70" font-size="11.5" text-anchor="middle" fill="var(--accent,#4f46e5)" font-weight="600">A</text>
<text x="308" y="272" font-size="11.5" text-anchor="middle" fill="var(--accent,#4f46e5)" font-weight="600">B</text>
<text x="185" y="292" font-size="11.5" text-anchor="middle" fill="currentColor" fill-opacity="0.7">이웃끼리는 최강, 먼 칩까지는 여러 번 건너간다</text>
<g stroke="currentColor" stroke-opacity="0.15" stroke-width="1.2">
<line x1="420" y1="250" x2="490" y2="120"/><line x1="465" y1="250" x2="490" y2="120"/>
<line x1="510" y1="250" x2="490" y2="120"/><line x1="555" y1="250" x2="490" y2="120"/>
<line x1="600" y1="250" x2="490" y2="120"/><line x1="645" y1="250" x2="490" y2="120"/>
<line x1="690" y1="250" x2="490" y2="120"/>
<line x1="420" y1="250" x2="620" y2="120"/><line x1="465" y1="250" x2="620" y2="120"/>
<line x1="510" y1="250" x2="620" y2="120"/><line x1="555" y1="250" x2="620" y2="120"/>
<line x1="600" y1="250" x2="620" y2="120"/><line x1="645" y1="250" x2="620" y2="120"/>
</g>
<polyline points="420,250 620,120 690,250" fill="none" stroke="var(--accent,#4f46e5)" stroke-width="3" stroke-linejoin="round"/>
<g fill="currentColor" fill-opacity="0.45">
<circle cx="465" cy="250" r="5"/><circle cx="510" cy="250" r="5"/><circle cx="555" cy="250" r="5"/>
<circle cx="600" cy="250" r="5"/><circle cx="645" cy="250" r="5"/>
</g>
<rect x="466" y="106" width="48" height="28" rx="7" fill="currentColor" fill-opacity="0.07" stroke="currentColor" stroke-opacity="0.3"/>
<rect x="596" y="106" width="48" height="28" rx="7" fill="var(--accent-soft,rgba(79,70,229,0.12))" stroke="var(--accent,#4f46e5)" stroke-opacity="0.5"/>
<text x="490" y="124" font-size="10.5" text-anchor="middle" fill="currentColor" fill-opacity="0.6">스위치</text>
<text x="620" y="124" font-size="10.5" text-anchor="middle" fill="var(--accent,#4f46e5)">스위치</text>
<circle cx="420" cy="250" r="8" fill="var(--accent,#4f46e5)"/>
<circle cx="690" cy="250" r="8" fill="var(--accent,#4f46e5)"/>
<text x="420" y="274" font-size="11.5" text-anchor="middle" fill="var(--accent,#4f46e5)" font-weight="600">A</text>
<text x="690" y="274" font-size="11.5" text-anchor="middle" fill="var(--accent,#4f46e5)" font-weight="600">B</text>
<text x="555" y="292" font-size="11.5" text-anchor="middle" fill="currentColor" fill-opacity="0.7">어디서 어디로 가든 스위치 두 번</text>
</svg>
<figcaption style="font-size:13px;color:var(--text-muted);margin-top:12px;line-height:1.65">왼쪽은 이웃끼리 주고받을 때 최강이고, 오른쪽은 어디서 어디로 가든 스위치 두 번이다. TPU v8i 실제 수치로는 1,024칩 기준 16홉이 7홉으로 줄었다.</figcaption>
</figure>

torus는 이웃끼리 주고받을 때 최강인데 MoE는 정반대로 전부가 전부에게 보낸다. 그러면 가장 먼 두 칩 사이 거리가 전체 속도를 정한다. 데이터센터 설계에서 bisection bandwidth만 보던 관성이 "거리와 완료 시간"으로 옮겨가고 있다는 신호로 읽힌다.

**느린 노드 관리가 네트워크 기능이 됐다.** TPU v8t의 새 네트워크는 "스케줄러가 느린 노드를 스텝 망치기 전에 죽이게 한다"를 목적으로 명시한다. 동기로 도는 작업에서는 제일 느린 노드가 전체 시간을 정하기 때문이다. AI 네트워크의 모니터링 요구사항은 시간 단위가 다르다는 뜻이다. 분 단위가 아니라 밀리초 이하다.

**표준 정치가 기술 선택을 앞선다.** UALink와 UEC는 둘 다 "구현이 아니라 표준을 소유한다"는 수다. 그런데 UALink는 스위치 칩이 2027년까지 없어서 Helios가 임시방편으로 나간다. 표준이 있어도 실리콘이 없으면 제품 일정이 표준을 안 기다려준다는 걸 그대로 보여준 사례다. 반대로 NVIDIA는 자기 연결 규격을 선택적으로 열기 시작했다. 개방 진영이 표준으로 포위하려 하자 폐쇄 진영이 문을 살짝 연 구도다.

**소프트웨어 해자는 네트워크에도 있다.** "NVIDIA는 고객사 안에 엔지니어를 보낸다"는 얘기가 계산에만 해당하는 게 아니다. NCCL은 단순한 라이브러리가 아니라 토폴로지를 아는 통신 구현체다. 새 네트워크가 성능을 내려면 통신 라이브러리가 그 구조를 이해해야 한다. 네트워크 경쟁은 링크 속도 경쟁이면서 동시에 통신 라이브러리 성숙도 경쟁이다.

## 12. 출처와 작성 노트

원문은 Jacob Peake의 ["AI Chip Architectures"](https://www.jepeake.com/ai-chip-architectures)다. NVIDIA, Google TPU, AMD, Cerebras 섹션과 Trainium의 구조·계산·메모리·통신 부분은 원문 본문을 직접 보고 정리했다.

다만 본문을 끝까지 확보하지 못했다. 추출이 Trainium 마지막 항목 중간에서 잘렸고 방식을 바꿔가며 시도해도 같은 지점에서 끊겼다. 그래서 두 부분은 공개 자료로 재구성했다.

- **Trainium의 연결·소프트웨어** — AWS 공식 문서 기반이다. 다만 "144칩 묶음, NeuronSwitch가 torus를 대체"는 원문 계보 표에서 확보한 내용이다.
- **Groq 섹션 전체** — Groq의 Hot Chips 발표 자료와 공개 문헌 기반이다. 원문이 확실히 말한 건 "200억 달러에 NVIDIA에 인수됐다"뿐이다.

원문에 결론 섹션이 더 있었을 수도 있는데 확인하지 못했다. 정확도가 중요한 용도라면 위 두 섹션은 원문을 직접 읽고 교차 확인하는 게 좋다.

그림은 전부 원문에 없는 것이고, 읽으면서 이해한 대로 내가 그렸다. 11장도 내 해석이다. 사실관계는 원문 위에 세웠지만 결론은 원저자의 것이 아니다.

LLMSO Week 5 Final Assignment로 정리했다.
