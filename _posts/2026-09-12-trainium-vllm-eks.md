---
layout: post
title: "Trainium에서 vLLM 서빙하기: 병목은 가속기가 아니었다"
date: 2026-09-12 10:00:00 +0900
categories: study llm-serving aws-trainium
tags: [LLM, vLLM, AWS, Trainium, EKS, Kubernetes, Neuron]
---

EKS 클러스터에 AWS Trainium 노드를 붙이고 vLLM으로 LLM을 서빙했다.
처리량과 지연을 측정하고, 오토스케일링과 배포를 시험하고, 토큰 단가를 계산했다.
NVIDIA에서 하던 감각이 그대로 통하지 않는 지점이 여러 군데 있었다.

측정 환경은 다음과 같다.

| 항목 | 값 |
| --- | --- |
| 클러스터 | EKS 1.33 |
| 노드 | `trn1.2xlarge` 1대 (NeuronCore-v2 2개, 가속기 메모리 32 GB) |
| 모델 | TinyLlama-1.1B-Chat-v1.0, bf16 |
| 서빙 | vLLM 0.9.0.dev, NeuronX Distributed, V0 엔진 |
| 컨텍스트 | `max_model_len` 1024, `max_num_seqs` 4 |
| 병렬화 | `tensor_parallel_size` 2, `enable_bucketing` false |

## 먼저 결론부터

- 처리량의 천장을 정한 것은 `max_num_seqs`다. 하드웨어가 아니다.
  이 값을 4에서 32로 올리면 처리량이 2.73배가 되고 p95 지연이 2.77배가 된다.
- `enable_bucketing`을 끄면 prefill이 입력 길이와 무관해진다.
  18 token 프롬프트와 802 token 프롬프트가 똑같이 48 ms를 쓴다.
- 용량 지표로 쓸 수 있는 게 거의 없다. 서버가 포화돼 지연이 3초가 된 순간에도
  CPU는 9%, NeuronCore 사용률은 83%다. 한가할 때도 81%다.
- 기본 Deployment 전략으로는 rollout이 조용히 멈춘다.
  요청은 100% 성공하는데 새 pod이 가속기를 잡지 못해 영원히 Pending이다.
- 추론만 한다면 Trainium을 살 이유가 없다.
  Inferentia2가 같은 처리량을 57% 가격에 내고, compile 결과물도 공유된다.
- 모델 크기의 상한은 31 GiB짜리 호스트가 정한다. 32 GB짜리 가속기가 아니다.

---

## 1. Trainium은 Kubernetes에서 어떻게 보이는가

노드의 allocatable resource를 보면 축이 두 개다.

```json
{
  "aws.amazon.com/neuron": "1",
  "aws.amazon.com/neuroncore": "2",
  "cpu": "7910m",
  "memory": "31315416Ki"
}
```

`trn1.2xlarge` 한 대가 Neuron device 1개이고 그 안에 NeuronCore-v2가 2개다.
NVIDIA라면 `nvidia.com/gpu: 1` 한 줄로 끝날 자리다.

이 차이가 vLLM 설정으로 이어진다. `tensor_parallel_size=2`가 가리키는 것은
칩 하나 안의 core 2개다. GPU 장 수를 세던 감각이 여기서는 맞지 않는다.

스케줄링도 다르다. Neuron runtime은 tensor parallelism을 쓸 때 연속된 core를 요구한다.
기본 Kubernetes scheduler는 "NeuronCore 2개"를 아무 core 2개로 보기 때문에
scheduler extension을 따로 깔고 pod이 그것을 쓰도록 지정해야 한다.

```bash
helm upgrade --install neuron-helm-chart oci://public.ecr.aws/neuron/neuron-helm-chart \
  --set "scheduler.enabled=true" --set "npd.enabled=false"
```

```yaml
spec:
  schedulerName: my-scheduler
```

GPU 쪽에는 없는 단계다.

## 2. 모델을 올리는 데 드는 비용

Trainium은 모델을 그냥 올려서 돌릴 수 없다. Neuron compiler로 한 번 변환해야 한다.
그래서 init container에서 미리 compile해 S3에 넣어두고 다음 pod이 재사용하는 구조가 된다.

compile 로그를 보면 graph가 두 개다.

```
neuronx-cc compile ... token_generation_model/_tp0_bk0/... --target=trn1 ... -O2
neuronx-cc compile ... context_encoding_model/_tp0_bk0/... --target=trn1 ... -O1
```

`context_encoding_model`이 prefill, `token_generation_model`이 decode다.
두 단계의 연산 성격이 다르니 최적화 레벨과 타일링도 다르게 잡힌다.
GPU에서는 같은 kernel이 두 단계를 다 처리하는데, Trainium은 AOT compile이라
단계별로 정적 graph를 미리 만들어 둔다.

cold start는 이렇게 나뉜다.

| 구간 | 소요 |
| --- | --- |
| init container (다운로드 + compile + S3 업로드) | 3분 32초 |
| vLLM API server 기동 | 34초 |
| 합계 | 4분 9초 |

여기에 더해 첫 배포에서는 container image pull에만 3분 가까이 더 걸렸다.
Neuron vLLM image가 크다.

### S3 cache에 실제로 들어가는 것

여기가 오해하기 쉽다. 버킷을 열어보면 이렇다.

```
  4.6 MiB  cache/model.pt
  6.1 KiB  cache/neuron_config.json
  1.5 MiB  cache/neuronxcc-2.20.9961.0+0acef03a/MODULE_56f0.../model.neff
  741 KiB  cache/neuronxcc-2.20.9961.0+0acef03a/MODULE_ae92.../model.neff
  ...
Total Objects: 11
Total Size: 9.8 MiB
```

9.8 MiB다. TinyLlama weights만 해도 2.2 GB인데 그게 없다.
들어 있는 것은 compile 결과물인 NEFF와 설정 파일뿐이다.
pod을 다시 띄우면 compile은 건너뛰지만 weights는 여전히 Hugging Face에서 받아온다.

그래서 warm restart는 이렇게 흐른다.

| 시각 | 사건 |
| --- | --- |
| 09:58:41 | pod 생성 |
| 09:58:43 | init container 시작하고 1초 안에 종료 (cache hit) |
| 09:58:45 | vLLM server container 시작 |
| 09:59:22 | `Application startup complete` |

41초다. compile 3분 32초가 1초 미만으로 줄었지만 weights 다운로드 시간은 그대로 남는다.

디렉터리 이름에 compiler 버전(`neuronxcc-2.20.9961.0+0acef03a`)이 들어간 것은 잘 만든 부분이다.
Neuron SDK를 올리면 cache가 자동으로 무효화된다.

실서비스라면 compile을 CI에서 굽고 weights도 S3나 EBS 스냅샷에 두는 게 맞다.
그러면 런타임은 읽기만 하면 된다.

## 3. 처리량의 천장은 설정이 정한다

요청 40개를 concurrency만 바꿔가며 돌렸다. 프롬프트는 짧고 `max_tokens=100`이다.

| Concurrency | 전체 소요 | Throughput | 평균 latency | p95 latency |
| --- | --- | --- | --- | --- |
| 1 | 24.12s | 1.66 req/s | 0.60s | 0.84s |
| 2 | 13.32s | 3.00 req/s | 0.65s | 0.97s |
| 4 | 7.01s | 5.70 req/s | 0.68s | 1.14s |
| 8 | 7.33s | 5.46 req/s | 1.36s | 1.99s |
| 16 | 6.42s | 6.23 req/s | 2.04s | 3.43s |

concurrency 4에서 처리량이 멈춘다. 그리고 그 숫자는 `max_num_seqs`와 같다.
vLLM이 한 batch에 sequence 4개까지만 넣으니 다섯 번째부터는 queue에서 기다린다.

1에서 4로 갈 때는 처리량이 3.4배 오르는데 평균 지연은 0.60초에서 0.68초로 거의 그대로다.
continuous batching이 일하고 있다는 뜻이다. batch에 자리가 있는 동안은 요청을 더 넣어도 거의 공짜다.
4를 넘기면 처리량은 고정되고 지연만 늘어난다.

### 왜 그런지는 칩 쪽 지표에서 드러난다

`neuron-monitor`로 NEFF 실행 지연을 보면 on-device decode step이
concurrency 1에서 6.53 ms, 4에서 6.53 ms, 16에서 6.54 ms다.
소수점 둘째 자리까지 같다.

step 하나에 6.53 ms인데 batch에 4개가 들어 있으면 그 시간에 token 4개가 나온다.
batch가 1이면 1개다. 처리량이 batch size에 비례하고 지연은 거의 안 변하는 이유가 이것이다.

vLLM이 보고한 TPOT은 9.21 ms였다. 칩에서 6.53 ms를 쓰고 나머지 2.7 ms가 host 쪽에서 나간다.
sampling, detokenize, scheduler 같은 것들이다.

### prefill과 decode의 비대칭

server가 스스로 보고한 지표다. 부하 시험 전체 트래픽 누적 기준이다.

| 지표 | 값 |
| --- | --- |
| 평균 end-to-end latency | 1.029 s |
| 평균 TTFT | 366 ms |
| 평균 TPOT | 9.21 ms |
| 평균 prefill 시간 | 48.96 ms |
| 평균 decode 시간 | 663.5 ms |
| Preemption 횟수 | 0 |

요청 하나가 1.029초 걸리는데 prefill이 49 ms고 decode가 663 ms다.
입력 260 token을 49 ms에 처리했으니 prefill은 약 5,300 tok/s,
decode는 token당 9.21 ms니까 108 tok/s다. 50배 차이다.

prefill은 병렬화가 되는 행렬 연산이고 decode는 token을 하나씩 만드는 순차 연산이다.
batching이 decode에서 특히 중요한 이유가 여기 있고, 그래서 `max_num_seqs`가 전체 처리량을 좌우한다.

`llmperf`로 측정한 token 단위 지표도 같은 그림이다.
입력 평균 256 token, 출력 평균 100 token, concurrency 5 조건이다.

| 지표 | 값 |
| --- | --- |
| TTFT 평균 | 251 ms |
| Inter-token latency 평균 | 12.03 ms |
| End-to-end latency 평균 | 1.175 s |
| 전체 output throughput | 336.6 tok/s |
| 실패 요청 | 0 |

## 4. prefill이 입력 길이와 무관하다

입력 길이만 바꿔가며 `max_tokens=1`로 테스트했다. prefill만 떼어내려는 것이다.

| prompt tokens | p50 latency | token당 |
| --- | --- | --- |
| 18 | 47.7 ms | 2.652 ms |
| 66 | 47.5 ms | 0.720 ms |
| 130 | 47.6 ms | 0.366 ms |
| 258 | 48.0 ms | 0.186 ms |
| 450 | 47.2 ms | 0.105 ms |
| 602 | 48.4 ms | 0.080 ms |
| 802 | 48.8 ms | 0.061 ms |

18 token이든 802 token이든 47~49 ms다. 44배 차이인데 지연은 3% 움직인다.

`enable_bucketing: false` 때문이다. bucketing을 끄면 NxD가 `max_model_len`에 맞춰
graph를 하나만 compile하고, 모든 요청이 그 1024짜리 graph를 padding해서 돈다.
18 token 프롬프트도 1024 token어치 연산을 한다.

token당 비용이 2.652 ms에서 0.061 ms로 43배 차이가 난다.
챗봇처럼 짧은 입력이 많은 워크로드라면 bucketing을 켜야 한다.
대신 여러 입력 길이에 대해 graph를 각각 compile하니 compile 시간과 cache 크기가 늘어난다.

설정 한 줄이 성능 특성을 통째로 바꾸는데, 이런 것은 문서만 읽어서는 보이지 않는다.

## 5. 오토스케일링에 쓸 지표가 없다

워크샵 기본 HPA는 CPU 사용률 70%를 기준으로 replica를 1에서 3 사이로 움직인다.
그런데 실제 추론 부하에서 CPU가 어떻게 움직이는지 확인해보면 이렇다.
concurrency 16으로 요청 300개를 계속 밀어 넣어 server를 완전히 포화시킨 상태다.

| 측정 대상 | 값 |
| --- | --- |
| 평균 latency | 2.95 s |
| p95 latency | 3.70 s |
| Throughput | 5.28 req/s (천장) |
| Pod CPU | 319~384 m (request 4000m 대비) |
| HPA가 읽은 값 | cpu: 9%/70% |
| Node CPU | 4% |

지연이 3초까지 늘어난 순간에도 HPA는 9%를 보고 있다. threshold 70%에는 근처도 못 간다.
연산은 Neuron 칩에서 일어나고 host CPU는 tokenizing과 scheduling 정도만 하기 때문이다.
이 HPA를 발동시키려면 pod 안에서 busy loop를 돌려야 한다.

그럼 NeuronCore 사용률을 보면 될 것 같지만 그쪽도 안 된다.

| Concurrency | NeuronCore 0 | NeuronCore 1 | Throughput | device memory |
| --- | --- | --- | --- | --- |
| 0 (idle) | 0.0% | 0.0% | | 3.63 GiB |
| 1 | 80.8% | 80.8% | 1.55 req/s | 3.63 GiB |
| 4 | 83.0% | 82.9% | 5.18 req/s | 3.63 GiB |
| 16 | 83.1% | 83.1% | 5.28 req/s | 3.63 GiB |

요청 하나만 보내도 core가 81%다. 거기서 처리량이 3.4배 오르는 동안 사용률은 2포인트 움직인다.
이 지표는 core가 뭔가를 실행 중인 시간의 비율이지 여유 용량이 아니다.
continuous batching이 decode step을 쉴 틈 없이 이어 붙이니 sequence가 하나든 넷이든 core는 계속 바쁘다.

device memory도 부하와 무관하게 고정이다. `max_model_len` 1024에 `max_num_seqs` 4면
KV cache가 미리 잡히고 그걸로 끝이다. 32 GB 중 11%만 쓴다.

움직이는 것은 queue 길이와 지연뿐이다. 그래서 그걸 봐야 한다.

### queue 길이를 HPA에 연결하기

`prometheus-adapter`로 vLLM의 queue 지표를 external metric으로 올린다.

```yaml
rules:
  default: false
  external:
    - seriesQuery: '{__name__="vllm:num_requests_waiting"}'
      resources: { namespaced: false }
      name: { as: "vllm_requests_waiting" }
      metricsQuery: "max(<<.Series>>)"
```

```yaml
metrics:
  - type: External
    external:
      metric: { name: vllm_requests_waiting }
      target: { type: Value, value: "2" }
```

concurrency 32로 부하를 걸면 이렇게 반응한다.

```
[12:12:58] waiting=0   running=0  hpa=0/2   replicas=1
[12:13:23] waiting=28  running=4  hpa=28/2  replicas=1
[12:13:47] waiting=28  running=4  hpa=28/2  replicas=2
[12:14:11] waiting=28  running=3  hpa=28/2  replicas=3
[12:17:02] waiting=0   running=0  hpa=0/2   replicas=3
```

25초 만에 반응했다. `running`이 4에서 멈춰 있는 것이 `max_num_seqs=4`고
나머지 28개가 `waiting`이다. 지표가 상태를 정확히 말해준다.
같은 부하에서 CPU 기준 HPA는 9%를 보고 있었다.

다만 신호를 고쳐도 용량이 생기지는 않는다. 늘어난 pod은 가속기가 없어 Pending이다.
Karpenter 같은 노드 프로비저너가 같이 있어야 의미가 있다.

## 6. 배포가 조용히 멈춘다

초당 5건씩 요청을 보내면서 `kubectl rollout restart`를 돌리고 실패를 셌다.
세 가지 상태로 각각 테스트했다.

| 시나리오 | 배포 전략 | probe | 실패율 | 장애 구간 | rollout 완료 |
| --- | --- | --- | --- | --- | --- |
| A | maxSurge 1 / maxUnavailable 0 (기본) | 없음 | 0% | 없음 | 안 됨 |
| B | maxSurge 0 / maxUnavailable 1 | 없음 | 43.8% | 95.2s | 됨 |
| C | maxSurge 0 / maxUnavailable 1 | 있음 | 23.2% | 64.7s | 됨 |

시나리오 A가 이 구조의 함정이다. 748건을 보내 하나도 실패하지 않았는데 배포가 끝나지 않는다.

```
vllm-deployment-85d9454bdb-x9rx8   0/1   Pending   2m8s
vllm-deployment-db754d957-thrlc    1/1   Running   117m
```

```
Waiting for deployment "vllm-deployment" rollout to finish: 1 old replicas are pending termination...
```

기본 전략은 새 pod이 Ready가 된 다음 기존 pod을 내린다.
그런데 Neuron device는 하나뿐이고 기존 pod이 잡고 있다.

```
0/1 nodes are available: 1 Insufficient aws.amazon.com/neuron, 1 Insufficient cpu,
1 Insufficient ephemeral-storage.
```

서비스는 멀쩡한데 배포만 멈춘다. CI에서 `rollout status`를 기다리고 있었다면 거기서 끝난다.
그리고 이 상태에서 답답해서 pod을 지우면 그때 장애가 난다.

`maxSurge: 0`으로 바꾸면 배포는 진행되지만 기존 pod을 먼저 죽인다.
95초 동안 요청의 44%가 connection refused로 실패했다.
그중 앞 49초는 Endpoints가 비어 있던 구간이고, 뒤 25초는 새 pod이 Ready로 표시됐는데
아직 서빙을 못 하던 구간이다.

뒤쪽 25초는 probe가 없어서 생긴 몫이다. 워크샵 매니페스트에는
readiness, liveness, startup probe가 하나도 없다.
vLLM은 `/health`를 이미 제공하니 몇 줄이면 된다.

```yaml
startupProbe:
  httpGet: { path: /health, port: 8080 }
  periodSeconds: 5
  failureThreshold: 60
readinessProbe:
  httpGet: { path: /health, port: 8080 }
  periodSeconds: 5
  failureThreshold: 3
```

붙이고 다시 재면 실패가 23.2%로 줄고 장애 구간이 64.7초가 된다.
Ready로 표시된 시각이 `Application startup complete`보다 2초 뒤다. 전에는 29초 앞섰다.

남은 64.7초는 probe로 못 줄인다. 가속기가 하나라 기존 pod이 죽어야 새 pod이 뜬다.
무중단으로 하려면 노드가 하나 더 있어야 한다.

여기서 가속기 워크로드의 용량 계획이 일반 워크로드와 다르다는 게 드러난다.
`trn1.2xlarge`는 8 vCPU이고 계정의 Trn 계열 한도도 8 vCPU다. 한 대가 천장이다.

```
Could not launch On-Demand Instances. VcpuLimitExceeded - You have requested more vCPU
capacity than your current vCPU limit of 8 allows for the instance bucket that the
specified instance type belongs to.
```

용량 계획은 계열별 vCPU 한도로 세워야 한다. 인스턴스 대수가 기준이 아니다.
오토스케일러를 붙여도 이 벽은 그대로다.

## 7. 과부하에서 깨지지 않는다

concurrency를 올려가며 어디서 5xx가 나는지 보려고 했다.

| Concurrency | Throughput | p50 | p95 | p99 | 실패 |
| --- | --- | --- | --- | --- | --- |
| 8 | 10.81 req/s | 0.59s | 1.43s | 1.67s | 0 |
| 32 | 10.71 req/s | 2.83s | 3.63s | 3.76s | 0 |
| 64 | 10.69 req/s | 5.69s | 6.61s | 6.94s | 0 |
| 128 | 10.93 req/s | 9.31s | 12.08s | 12.50s | 0 |
| 256 | 10.94 req/s | 9.09s | 17.35s | 17.84s | 0 |

256까지 올려도 에러가 0이다. 처리량은 10.8 req/s에 고정되고 지연만 늘어난다.
p99가 1.67초에서 17.84초가 됐다.

vLLM에는 admission control이 없다. 들어오는 요청을 전부 받아 queue에 쌓는다.
클라이언트 쪽에 backpressure 신호가 가지 않는다.
타임아웃 5초짜리 클라이언트라면 진작 포기했을 요청을 서버는 계속 계산한다.
그 계산은 아무도 안 기다린다.

앞에 뭔가를 둬야 한다. ingress에서 `limit_conn`이나 `limit_req`로 동시 연결을 막든,
gateway에서 queue 깊이를 보고 429를 돌려주든.

### 그래서 알림도 error rate로 걸면 안 된다

네 가지 규칙을 넣고 부하를 걸어봤다.

| 규칙 | 조건 | for |
| --- | --- | --- |
| VLLMQueueBacklog | `vllm:num_requests_waiting > 2` | 1m |
| VLLMHighTTFT | TTFT p95 > 1s | 2m |
| VLLMPodPending | vLLM pod이 Pending | 2m |
| VLLMNoReadyPod | Ready인 pod이 0개 | 1m |

```
[12:22:28] (없음)
[12:22:48] QueueBacklog:pending  HighTTFT:pending  PodPending:pending
[12:23:48] QueueBacklog:firing   HighTTFT:pending  PodPending:pending
[12:24:48] QueueBacklog:firing   HighTTFT:firing   PodPending:firing
```

그 5분 동안 요청 1600건이 전부 성공했다. 실패 0건이다.
error rate 기준으로 알림을 걸어놨다면 아무 일도 없었던 것처럼 지나간다.
사용자는 6초씩 기다리는데 대시보드는 초록색이다.

`VLLMPodPending`은 앞에서 본 멈춘 배포도 잡는다.
그 상황도 요청은 100% 성공하고 있었으니 다른 지표로는 안 보인다.

vLLM이 내보내는 지표는 충분하다. 쓰기만 하면 된다.

```
vllm:num_requests_running          vllm:num_requests_waiting
vllm:time_to_first_token_seconds   vllm:time_per_output_token_seconds
vllm:request_prefill_time_seconds  vllm:request_decode_time_seconds
vllm:e2e_request_latency_seconds   vllm:num_preemptions_total
```

함정이 하나 있다. `vllm:gpu_cache_usage_perc`는 Neuron backend에서 항상 0이다.
이름에 `gpu`가 박혀 있는 데서 알 수 있듯 CUDA backend 전용 계측이다.
KV cache 압박은 `num_requests_waiting`과 `num_preemptions_total`로 봐야 한다.

## 8. 단가를 움직이는 두 개의 레버

`trn1.2xlarge` 온디맨드가 시간당 1.34달러다. 초당 0.000372달러다.
이 하드웨어에서 1M output token을 만드는 데 얼마가 드는지 계산해보면
같은 칩에서도 조건에 따라 네 배 차이가 난다.

| 조건 | output throughput | 1M token당 |
| --- | --- | --- |
| concurrency 1 | 84.7 tok/s | 4.39 달러 |
| concurrency 5 | 336.6 tok/s | 1.11 달러 |

batching을 얼마나 채우느냐가 그대로 단가다. 그럼 더 채우면 된다.

### 레버 하나: max_num_seqs

device memory가 32 GB 중 3.63 GiB밖에 안 쓰고 있으니 올릴 여지가 있다.
값마다 S3 cache를 비우고 재compile하면서 4에서 32까지 올려봤다.
부하는 매번 `max_num_seqs`의 4배 concurrency로 걸었다.

| max_num_seqs | Throughput | output tok/s | 평균 latency | p95 | device memory | compile | $/1M |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 5.10 req/s | 248 | 3.00s | 3.78s | 3.63 GiB | 214s | 1.50 |
| 8 | 7.73 req/s | 377 | 3.99s | 4.77s | 3.72 GiB | 217s | 0.99 |
| 16 | 10.63 req/s | 518 | 5.80s | 6.88s | 3.89 GiB | 221s | 0.72 |
| 32 | 13.92 req/s | 678 | 8.89s | 10.48s | 4.24 GiB | 229s | 0.55 |

4에서 32로 가면 처리량이 2.73배가 되고 단가가 2.73배 떨어진다.
그리고 p95 지연이 2.77배 나빠진다. 거의 1대1 교환이다.
공짜가 없다. 지연을 팔아 비용을 산다.

몇 가지가 눈에 띈다. device memory는 3.63에서 4.24 GiB로 0.6 GiB밖에 안 늘었다.
batch를 8배로 키웠는데 KV cache가 이만큼만 는다. `max_model_len`이 1024라 그렇다.
메모리는 한계가 아니었다.

compile 시간은 214초에서 229초로 거의 안 변한다. batch size는 compile 비용에 영향이 없다.
실패는 네 값 모두 0건이다.

### 레버 둘: 인스턴스 계열

`inf2.xlarge` 노드를 하나 붙여서 같은 모델, 같은 설정으로 나란히 테스트했다.
두 칩의 가속기 쪽 사양은 동일하다.

| | trn1.2xlarge | inf2.xlarge |
| --- | --- | --- |
| Neuron device | 1 | 1 |
| NeuronCore | 2 | 2 |
| 가속기 메모리 | 32 GB | 32 GB |
| 호스트 vCPU | 7910m | 3920m |
| 호스트 메모리 | 31 GB | 15 GB |
| 시간당 | $1.34 | $0.76 |

성능은 이렇게 나왔다.

| Concurrency | trn1.2xlarge | inf2.xlarge |
| --- | --- | --- |
| 1 | 1.54 req/s, 0.65s | 1.40 req/s, 0.71s |
| 4 | 4.92 req/s, 0.78s | 5.08 req/s, 0.77s |
| 16 | 5.68 req/s, 2.25s | 5.68 req/s, 2.28s |

노이즈 범위다. 같은 NeuronCore-v2 세대이니 당연하기도 하다.
그런데 가격은 trn1이 1.76배 비싸다.
Trainium은 학습용이라 inter-chip interconnect와 넉넉한 호스트 자원에 값이 붙어 있는데
추론만 하면 그걸 안 쓴다.

compile 결과물도 공유된다. Trainium에서 구운 NEFF를 Inferentia2 pod이 그대로 읽었다.

```
INFO [neuronx_distributed.py:244] Successfully loaded precompiled model artifacts from /shared/model/cache
```

S3 cache 하나로 두 종류 노드를 채울 수 있다는 뜻이다.
가속기 계열별로 quota가 따로 잡히니, 한 계열이 막혔을 때 다른 계열로 용량을 채우는 길이 된다.

### 두 레버를 합치면

워크샵 기본값은 `trn1.2xlarge`에 `max_num_seqs=4`다. 1M output token당 1.50달러다.
`inf2.xlarge`에 `max_num_seqs=32`면 0.31달러다. 4.8배 차이다.

하드웨어도 모델도 그대로다. 인스턴스 종류 하나와 설정값 하나를 바꿨을 뿐이다.
그 대신 p95 지연이 3.78초에서 10.48초가 된다. SLO가 허락하는 만큼만 가져가면 된다.

## 9. 모델 크기의 상한은 호스트가 정한다

여기까지는 전부 1.1B 모델이다. 모델을 키우면 무엇이 먼저 막히는지 테스트했다.

`trn1.2xlarge`의 Neuron device에는 32 GB가 있다. 8B도 bf16으로 들어간다.
그런데 그 weights를 칩에 올리려면 호스트가 먼저 들고 샤딩해야 한다.

Qwen3-4B 체크포인트는 12 GB이고 `torch_dtype`은 `bfloat16`이다.
로딩 시작 1분 만에 python 프로세스의 RSS가 8.6 GB가 되고 계속 자라 28 GB를 넘긴다.
체크포인트 크기의 세 배 가까이 쓴다. 로그에 단서가 있다.

```
UserWarning: Found torch.float32 weights in checkpoint: lm_head.weight. Will convert to torch.bfloat16
```

NxD가 로딩 과정에서 fp32를 거치고, tensor parallel 2로 나누면서 사본을 더 만든다.

| 모델 | 체크포인트 | 결과 |
| --- | --- | --- |
| TinyLlama-1.1B | 2.2 GB | 정상. device memory 3.63 GiB |
| Qwen3-4B | 12 GB | 호스트 28Gi 초과로 OOMKilled |
| Qwen3-8B | 16 GB | 메모리 제한이 없으면 노드 전체가 다운 |

`trn1.2xlarge`에서 실제로 올릴 수 있는 모델 상한은 1.1B와 4B 사이 어딘가다.
32 GB짜리 가속기를 달고도 그렇다. 병목은 호스트 31 GiB다.

큰 모델을 Neuron에 올리려면 가속기 개수만 보면 안 된다.
`trn1.32xlarge`처럼 호스트 메모리가 512 GB인 인스턴스를 고르거나,
체크포인트를 미리 샤딩해서 로딩 피크를 낮추는 경로가 필요하다.

### 메모리 제한이 없으면 노드가 죽는다

워크샵 매니페스트의 resources는 이렇다.

```yaml
resources:
  limits:
    aws.amazon.com/neuron: 1
    ephemeral-storage: 50Gi
    cpu: "8000m"
  requests:
    aws.amazon.com/neuron: 1
    ephemeral-storage: 50Gi
    cpu: "4000m"
```

`memory`가 없다. request도 limit도 없다.
가속기, 디스크, CPU는 선언하는데 호스트 메모리만 빠져 있다.

제한 없이 8B를 올리면 컨테이너가 호스트 메모리를 다 먹을 때까지 자라고,
6분 뒤 kubelet이 상태 보고를 멈춘다.

```
MemoryPressure   Unknown   NodeStatusUnknown   Kubelet stopped posting node status.
MountVolume.SetUp failed for volume "s3-model-cache-pv" :
  dial unix /var/lib/kubelet/plugins/s3.csi.aws.com/csi.sock: connect: no such file or directory
```

pod 하나가 감당 못 할 모델을 집었을 뿐인데 노드 전체가 스케줄 불가 상태가 된다.
복구하려면 노드그룹을 0으로 줄였다가 다시 올려 인스턴스를 교체해야 한다.

limit을 걸면 같은 일이 이렇게 끝난다.

```
exitCode: 137
reason: OOMKilled
restartCount: 1
```

컨테이너만 죽고 쿠버네티스가 재시작한다. 노드는 Ready 그대로다.
가속기 노드에서 호스트 메모리는 아무도 선언하지 않는 자원이고, 터지면 노드가 통째로 나간다.

## 10. 실서비스로 가져간다면

측정한 것들을 그대로 체크리스트로 옮기면 이렇다.

| 항목 | 흔한 기본값 | 바꿀 것 | 근거 |
| --- | --- | --- | --- |
| 오토스케일 지표 | CPU 사용률 | `vllm:num_requests_waiting` 또는 TTFT SLO | 5장 |
| 노드 확장 | 없음 | Karpenter로 Neuron node pool | 5장 |
| 배포 전략 | maxSurge 1 / maxUnavailable 0 | 명시적으로 고를 것. 무중단은 N+1 용량이 필요 | 6장 |
| probe | 없음 | `/health`로 startup·readiness·liveness | 6장 |
| 과부하 방어 | 없음 | ingress나 gateway에서 동시 연결과 queue 깊이 제한 | 7장 |
| 알림 | error rate | queue 길이, TTFT, Pending pod | 7장 |
| `max_num_seqs` | 보수적인 기본값 | SLO가 허락하는 만큼 올릴 것 | 8장 |
| 인스턴스 계열 | 관행대로 | 추론이면 Inferentia | 8장 |
| memory limit | 없음 | 컨테이너마다 지정 | 9장 |
| compile 캐시 | 런타임 compile | CI에서 굽고 컴파일러 버전 고정 | 2장 |
| weights | 매번 Hugging Face | S3나 EBS 스냅샷에 두고 예열 | 2장 |
| secret | ConfigMap에 평문 | Secret 또는 외부 secret store | |

마지막 항목만 부연한다. 워크샵 ConfigMap에는 Hugging Face token이 평문으로 들어간다.
Secret을 따로 만들어 놓고도 ConfigMap에 같은 값을 또 넣는 구성이다.
ConfigMap은 RBAC 범위가 넓고 `kubectl describe`에도 그대로 나온다.

## 11. 마무리

기억에 남는 숫자는 두 개다.

하나는 HPA가 읽은 9%다. 서버가 완전히 포화돼 지연이 3초까지 늘어난 순간에도
CPU 지표는 한 자릿수였다. CPU 기준 오토스케일링은 가속기 워크로드에서 동작하지 않는다.

다른 하나는 prefill의 48 ms다. 입력이 18 token이든 802 token이든 같았다.
`enable_bucketing` 한 줄이 성능 특성을 통째로 바꿔놓는다.

설정값 하나가 하드웨어보다 크게 작용하는 장면을 여러 번 봤다.
처리량 천장을 정한 것은 `max_num_seqs`였고,
1M token 단가를 4.8배 바꾼 것도 인스턴스 계열과 그 값이었다.
그리고 모델 크기의 상한을 정한 것은 31 GiB짜리 호스트였다. 32 GB짜리 가속기가 아니었다.

가속기가 바뀌면 병목도 바뀐다. GPU에서 쓰던 대시보드와 오토스케일 규칙을
그대로 옮기면 대부분 동작하지 않는다. 무엇을 봐야 하는지부터 다시 정해야 한다.

다음에 볼 것을 세 가지 적어둔다.
`enable_bucketing`을 켰을 때 prefill이 얼마나 싸지고 compile 시간이 얼마나 늘어나는지,
`max_num_seqs`를 32보다 더 올리면 어디서 꺾이는지,
그리고 체크포인트를 미리 샤딩해두면 로딩 피크가 얼마나 내려가는지다.

---

## 참고 자료

- [AWS Neuron 문서](https://awsdocs-neuron.readthedocs-hosted.com/)
- [vLLM User Guide for NxD Inference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide-v1.html)
- [vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron)
- [aws-neuron/aws-neuron-eks-samples](https://github.com/aws-neuron/aws-neuron-eks-samples)
- [ray-project/llmperf](https://github.com/ray-project/llmperf)
- [Mountpoint for Amazon S3 CSI Driver](https://github.com/awslabs/mountpoint-s3-csi-driver)
