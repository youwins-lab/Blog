---
title: "Trainium 위의 vLLM: EKS에서 LLM 서빙하고 측정한 기록"
date: 2026-09-12
tags: [LLM, vLLM, AWS, Trainium, EKS, Kubernetes, Neuron]
---

# Trainium 위의 vLLM

> AWS Trainium 가속기를 붙인 EKS 클러스터에 vLLM을 올리고, 서빙 성능과 운영 특성을 측정한 기록이다.

클러스터는 EKS 1.33이고 노드는 `trn1.2xlarge` 한 대, 모델은 TinyLlama-1.1B-Chat-v1.0이다.
AWS 워크샵 환경("Scaling LLM Inference with vLLM and AWS Trainium")에서 진행했다.
EKS API endpoint가 private 전용이라 모든 명령은 VPC 안쪽의 워크샵 EC2 인스턴스에서 실행했다.

랩 자체는 배포까지만 다룬다. 그래서 랩을 끝낸 뒤에 네 갈래로 더 들어갔다.
가속기가 실제로 얼마나 일하는지(9장), 배포와 과부하와 알림 같은 운영 측면(10장),
용량과 비용을 움직이는 레버(11장), 그리고 모델을 키웠을 때 무엇이 먼저 막히는지(12장)다.

### 먼저 결론부터

- 워크샵 values 파일 그대로는 Prometheus와 Alertmanager가 기동하지 않는다.
  scrape job 이름 충돌과 폐기된 chart key 때문이다. (8장)
- NeuronCore 사용률은 용량 지표로 못 쓴다. 요청 하나만 보내도 81%고 완전히 포화돼도 83%다. (9.1절)
- `enable_bucketing`을 끄면 prefill이 입력 길이와 무관해진다.
  18 token 프롬프트와 802 token 프롬프트가 똑같이 48 ms를 쓴다. (9.3절)
- CPU 기준 HPA는 발동하지 않는다. 서버가 포화돼 latency가 3초가 된 순간에도 9%를 보고 있었다. (7.2절)
- 기본 배포 전략으로는 rollout이 조용히 멈춘다. 요청은 100% 성공하는데 배포만 끝나지 않는다. (10.1절)
- 추론만 한다면 Trainium을 살 이유가 없다. Inferentia2가 같은 처리량을 57% 가격에 낸다. (11.4절)
- `max_num_seqs`를 4에서 32로 올리면 단가가 2.73배 내려가고 p95 latency가 2.77배 올라간다. (11.5절)
- 모델 크기의 상한은 31 GiB짜리 호스트가 정한다. 32 GB짜리 가속기가 아니다.
  그리고 컨테이너에 memory limit이 없으면 pod이 아닌 노드가 죽는다. (12장)

---

## 1. 전체 그림

```
내 노트북
     │ SSH
     ▼
워크샵 EC2 (t3.2xlarge, VPC 안쪽)
     │ kubectl / eksctl / helm
     ▼
EKS 1.33  ai-infra-summit-test-cluster  (private endpoint)
     │
     ├── managed node group  neuron-trn1-2x : trn1.2xlarge × 1
     │      └── Neuron device 1개 = NeuronCore-v2 2개
     │
     ├── neuron-device-plugin  : Neuron device를 K8s allocatable resource로 노출
     ├── k8s-neuron-scheduler  : NeuronCore 할당을 담당하는 scheduler extension
     ├── my-scheduler          : 위 extension을 사용하는 별도 scheduler
     └── s3-csi-driver         : S3 bucket을 PV로 mount (compile cache 공유용)

vLLM pod
     ├── initContainer model-prep : HF에서 모델 받아 Neuron용으로 compile → S3에 cache
     └── container vllm-server    : compile된 artifact로 OpenAI 호환 API 서빙
```

Trainium은 GPU처럼 모델을 그냥 올려서 돌릴 수가 없다.
Neuron compiler로 한 번 변환해야 하고, 이게 몇 분씩 걸린다.
그래서 init container에서 미리 compile해 S3에 넣어두고 다음 pod이 재사용하는 구조다.

---

## 2. Lab 1 — node group 만들고 Neuron 스택 올리기

### 2.1 도구부터

워크샵 EC2에는 `kubectl`만 있고 AWS CLI도 Helm도 없다. 이것부터 채웠다.

```bash
sudo apt-get install -y python3-pip jq unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip -q awscliv2.zip && sudo ./aws/install --update
curl -sS https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

깔고 나니 aws-cli 2.36.43, helm v3.22.0, eksctl 0.230.0, kubectl v1.37.0이 잡혔다.
3분쯤 걸렸다.

### 2.2 node group

worker AMI는 하드코딩하지 않고 SSM parameter에서 조회해 온다.

```bash
export WORKER_AMI=$(aws ssm get-parameter \
  --name /aws/service/eks/optimized-ami/1.33/amazon-linux-2023/x86_64/neuron/recommended/image_id \
  --region us-west-2 --query "Parameter.Value" --output text)
```

`ami-0e08c07b0376ba3f8`이 나왔다. 워크샵 architecture 페이지에는
`ami-08695d32a8bb6c5a5`라고 적혀 있다. 문서 쪽이 오래된 것 같아서 조회값을 그대로 썼다.

그리고 이건 모르면 삽질할 수 있는 부분인데, `trn1.2xlarge`가 모든 AZ에 있지는 않다.

```bash
aws ec2 describe-instance-type-offerings --location-type availability-zone \
  --filters "Name=instance-type,Values=trn1.2xlarge" \
  --query 'InstanceTypeOfferings[*].Location' --output text
```

```
us-west-2d  us-west-2b
```

둘뿐이다. us-west-2a와 us-west-2c에는 없다.
아무 subnet이나 썼으면 capacity 부족으로 실패했을 텐데, 워크샵 스크립트가
지원 AZ를 먼저 조회하게 짜여 있어서 걸려들지 않았다. 실제 node는 us-west-2b에 떴다.

```bash
eksctl create nodegroup --config-file=eks_nodegroup.yaml
```

CloudFormation stack 생성부터 node Ready까지 2분 41초. 예상보다 빨랐다.
eksctl이 Neuron AMI를 알아보고 `neuron-device-plugin` DaemonSet까지 알아서 깔아줬다.
어떤 조건으로 그렇게 판단하는지는 확인하지 않았다.

```
NAME                                       STATUS   VERSION                INSTANCE
ip-10-0-5-185.us-west-2.compute.internal   Ready    v1.33.13-eks-cb19647   trn1.2xlarge
```

### 2.3 Trainium은 Kubernetes에 어떻게 보이나

node의 allocatable resource를 봤다.

```json
{
  "aws.amazon.com/neuron": "1",
  "aws.amazon.com/neuroncore": "2",
  "cpu": "7910m",
  "memory": "31315416Ki",
  "ephemeral-storage": "95491281146",
  "pods": "58"
}
```

`trn1.2xlarge` 한 대가 Neuron device 1개, 그 안에 NeuronCore-v2 2개다.
NVIDIA였다면 `nvidia.com/gpu: 1` 한 줄로 끝날 자리에 축이 두 개다.

이게 나중에 걸린다. vLLM에서 `tensor-parallel-size=2`를 주면
그건 GPU 1장이 아니고 칩 하나 안의 core 2개를 뜻한다. 같은 파라미터인데 물리적 의미가 다르다.

### 2.4 Neuron 스택

워크샵은 eksctl이 자동으로 깔아준 device plugin을 지우고 공식 Helm chart로 다시 깐다.
버전 관리를 Helm에 맡기려는 의도인 듯한데, 문서에 이유가 적혀 있지는 않다.

```bash
kubectl delete daemonset neuron-device-plugin -n kube-system
kubectl delete clusterrole neuron-device-plugin
kubectl delete serviceaccount neuron-device-plugin -n kube-system
kubectl delete clusterrolebinding neuron-device-plugin

helm upgrade --install neuron-helm-chart \
  oci://public.ecr.aws/neuron/neuron-helm-chart --set "npd.enabled=false"
```

그다음이 Neuron scheduler extension이다.

```bash
helm upgrade --install neuron-helm-chart oci://public.ecr.aws/neuron/neuron-helm-chart \
  --set "scheduler.enabled=true" --set "npd.enabled=false"
```

기본 Kubernetes scheduler는 NeuronCore 2개를 요청받으면 아무 core 2개나 주면 된다고 본다.
그런데 Neuron runtime은 tensor parallelism을 쓸 때 연속된 core를 요구한다고 한다.
extension이 그 제약을 반영해 배치를 결정한다. pod 두 개가 뜬다.

```
kube-system   k8s-neuron-scheduler-785c8d99f8-hdklt   1/1   Running
kube-system   my-scheduler-55f56bc9f8-rctzh           1/1   Running
```

그래서 나중에 vLLM manifest에 `schedulerName: my-scheduler`가 들어간다.
GPU 쪽에는 없는 고민이다.

### 2.5 S3 CSI driver와 cache bucket

compile 결과를 담을 bucket을 만들고, S3를 PV로 mount할 수 있게 CSI driver를 깐다.

```bash
aws s3 mb "s3://ai-infra-summit-vllm-models-cache-<account-id>" --region us-west-2

helm repo add aws-mountpoint-s3-csi-driver https://awslabs.github.io/mountpoint-s3-csi-driver
helm upgrade --install aws-mountpoint-s3-csi-driver \
  --namespace kube-system aws-mountpoint-s3-csi-driver/aws-mountpoint-s3-csi-driver
```

Mountpoint for Amazon S3 CSI Driver v2.8.0이 올라갔다.
여기까지가 Lab 1이고, cluster 상태는 이렇다.

```
kube-system   aws-node-vj2hn                          2/2   Running
kube-system   coredns-75cb89d95b-knpdx                1/1   Running
kube-system   coredns-75cb89d95b-mc86t                1/1   Running
kube-system   k8s-neuron-scheduler-785c8d99f8-hdklt   1/1   Running
kube-system   kube-proxy-nd6fg                        1/1   Running
kube-system   my-scheduler-55f56bc9f8-rctzh           1/1   Running
kube-system   neuron-device-plugin-l6lpd              1/1   Running
kube-system   s3-csi-controller-5df587766f-cc6l5      1/1   Running
kube-system   s3-csi-node-mpfzn                       3/3   Running
```

여기까지는 순조로웠다.

---
## 3. Lab 2 — vLLM 배포

### 3.1 뭘 만드는가

Secret 하나, ConfigMap 하나, S3를 물고 있는 PV와 PVC, Deployment, LoadBalancer Service.
다섯 개다. Deployment 안에 init container와 vLLM server container가 같이 들어간다.

ConfigMap에 들어가는 값 중 중요한 것들이다.

| 키 | 값 | 의미 |
| --- | --- | --- |
| MODEL_NAME | tinyLlama/TinyLlama-1.1B-Chat-v1.0 | 데모용 1.1B 모델 |
| TENSOR_PARALLEL_SIZE | 2 | NeuronCore 2개에 모델 분할 |
| MAX_MODEL_LEN | 1024 | context length 상한 |
| MAX_NUM_SEQS | 4 | 동시에 처리할 sequence 수 |
| NEURON_RT_VISIBLE_CORES | 0-1 | 사용할 core 범위 |
| VLLM_NEURON_FRAMEWORK | neuronx-distributed-inference | NxD backend |
| enable_bucketing | false | 입력 길이별 bucketing 비활성화 |

`MAX_NUM_SEQS`가 4다. 이 숫자는 나중에 benchmark에서 그대로 튀어나온다.

### 3.2 Neuron compile이 하는 일

init container 로그에 compiler 호출이 찍힌다.

```
neuronx-cc compile --framework=XLA .../token_generation_model/_tp0_bk0/model....hlo_module.pb
  --output .../model....neff --target=trn1 --auto-cast=none --model-type=transformer
  --tensorizer-options=--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=1
  --vectorize-strided-dma --lnc=1 -O2
```

```
neuronx-cc compile --framework=XLA .../context_encoding_model/_tp0_bk0/model....hlo_module.pb
  --output .../model....neff --target=trn1 ... --cc-pipeline-tiling-factor=2 ... -O1
```

graph가 두 개다. `context_encoding_model`이 prefill, `token_generation_model`이 decode다.
prefill은 `-O1`에 tiling 2, decode는 `-O2`에 tiling 1로 다르게 잡혀 있다.
두 단계의 연산 성격이 다르니 최적화 방향도 다르게 잡힌다.

GPU에서는 같은 kernel이 두 단계를 다 처리한다.
Trainium은 AOT compile이라 단계별로 정적 graph를 미리 만들어 둔다.
compile 비용은 요청마다 드는 게 아니고 배포 때 한 번 든다.

### 3.3 얼마나 걸렸나

pod status에서 뽑은 타임스탬프다.

| 구간 | 시각 | 소요 |
| --- | --- | --- |
| pod 생성 | 15:30:20 | |
| init container 시작 | 15:30:22 | |
| init container 종료 (exit 0) | 15:33:54 | 3분 32초 |
| vLLM server container 시작 | 15:33:55 | |
| API route 등록 완료 | 15:34:29 | 34초 |
| 합계 | | 약 4분 9초 |

단 이건 container image가 이미 node에 받아진 상태의 숫자다.
맨 처음에는 image pull에만 3분 가까이 더 걸렸다. Neuron vLLM image가 꽤 크다.

### 3.4 S3 cache에 뭐가 올라갔나

bucket을 열어봤다.

```
  4.6 MiB  cache/model.pt
  6.1 KiB  cache/neuron_config.json
  547 KiB  cache/neuronxcc-2.20.9961.0+0acef03a/MODULE_56f0.../model.hlo_module.pb
  1.5 MiB  cache/neuronxcc-2.20.9961.0+0acef03a/MODULE_56f0.../model.neff
  1.6 MiB  cache/neuronxcc-2.20.9961.0+0acef03a/MODULE_56f0.../wrapped_neff.hlo
  851 KiB  cache/neuronxcc-2.20.9961.0+0acef03a/MODULE_ae92.../model.hlo_module.pb
  741 KiB  cache/neuronxcc-2.20.9961.0+0acef03a/MODULE_ae92.../model.neff
  ...
Total Objects: 11
Total Size: 9.8 MiB
```

9.8 MiB. TinyLlama weights만 해도 2.2GB인데 그게 통째로 빠져 있다.
여기 들어 있는 건 compile 결과물인 NEFF와 설정 파일뿐이다.
pod을 다시 띄우면 compile은 건너뛰지만 weights는 여전히 Hugging Face에서 받아온다.

워크샵 본문은 "subsequent runs use the S3 model cache and take only ~20 seconds"라고 한다.
weights 다운로드 시간이 그대로 남으니 그 말을 곧이곧대로 믿으면 안 된다.
실측은 41초다. 9.4절에 있다.

저장 경로도 ConfigMap의 `S3_PREFIX: compiled-models`가 아니고 그냥 `cache/`다.
그 변수는 어디에서도 쓰이지 않는 것 같다.

디렉터리 이름에 compiler 버전(`neuronxcc-2.20.9961.0+0acef03a`)이 들어간 건 잘 만든 부분이다.
Neuron SDK를 올리면 cache가 자동으로 무효화된다.

### 3.5 첫 추론

```bash
curl -s http://localhost:8080/v1/models | jq '.data[0] | {id, max_model_len}'
```

```json
{ "id": "tinyLlama/TinyLlama-1.1B-Chat-v1.0", "max_model_len": 1024 }
```

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"tinyLlama/TinyLlama-1.1B-Chat-v1.0",
       "messages":[{"role":"user","content":"Hello, how are you?"}],
       "max_tokens":100,"temperature":0.7}'
```

100 token에 1.84초. 초당 54 token쯤 된다.

server 로그에서 하나 더 봤다.

```
WARNING [arg_utils.py:1611] device type=neuron is not supported by the V1 Engine. Falling back to V0.
```

vLLM V1 engine이 Neuron을 아직 지원하지 않아 V0로 내려간다.
V1의 scheduler 개선이나 chunked prefill을 Trainium에서는 못 쓴다.
실서비스를 검토한다면 알고 있어야 할 제약이다.

---

## 4. Lab 3 — Ingress

NGINX Ingress Controller를 Helm으로 깔고 path 기반 routing을 붙인다.

```bash
helm upgrade --install ingress-nginx ingress-nginx \
  --repo https://kubernetes.github.io/ingress-nginx \
  --namespace ingress-nginx --create-namespace
```

Ingress가 NLB 주소를 받는 데까지 49초 걸렸고, 첫 요청부터 200이 떨어졌다.

```
NAMESPACE   NAME                  CLASS   ADDRESS                                       PORTS
default     vllm-ingress-simple   nginx   a3008b3e...us-west-2.elb.amazonaws.com        80
```

기왕 붙인 김에 모델한테 자기가 돌아가는 칩에 대해 물어봤다.

```bash
curl -X POST "http://$VLLM_INGRESS/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"tinyLlama/TinyLlama-1.1B-Chat-v1.0",
       "messages":[{"role":"user","content":"Explain what AWS Trainium is in two sentences."}],
       "max_tokens":120}'
```

> AWS Trainium is an AWS Training and Certification management platform that provides
> training materials, e-learning courses, certification trackers, and certification exams
> for AWS services.

Trainium은 ML 가속기 칩이다. 교육 플랫폼이 아니다.
이름만 보고 지어낸 티가 역력하다. 1.1B 모델이니까 그러려니 하고 넘어갔다.

한 가지 걸리는 건 이 랩이 끝나면 load balancer가 두 개가 된다는 점이다.
`vllm-service`가 LoadBalancer 타입이라 이미 ELB를 하나 만들어 뒀는데
Ingress Controller가 또 하나를 만든다. 실서비스라면 Service는 ClusterIP로 두는 게 맞다.

---
## 5. Lab 4 — Prometheus가 두 가지 이유로 안 떴다

Prometheus와 Grafana를 Helm으로 올리는 단계다. 워크샵이 준 values 파일 그대로 돌렸다.

```
NAME                        READY   STATUS             RESTARTS
prometheus-alertmanager-0   0/1     Pending            0
prometheus-server           1/2     CrashLoopBackOff   1
```

둘 다 죽었는데 원인이 서로 달랐다.

### 5.1 job 이름이 겹쳤다

```
level=ERROR msg="Error loading config (--config.file=/etc/config/prometheus.yml)"
err="parsing YAML file /etc/config/prometheus.yml:
     found multiple scrape configs with job name \"kubernetes-pods\""
```

워크샵의 `prometheus-values.yaml`이 `kubernetes-pods`라는 이름으로 scrape job을 추가한다.
prometheus-community chart에 이미 같은 이름의 기본 job이 있다.
Prometheus는 job 이름 중복을 허용하지 않아서 config 로드에서 바로 죽는다.

job 이름을 `annotated-pods`로 바꾸니 떴다.
나중에 보니 chart 기본 job이 이미 annotation 기반 pod discovery를 하고 있어서
이 job은 처음부터 없어도 됐던 것 같다.

### 5.2 Alertmanager는 chart key가 바뀐 탓이었다

```
Warning  FailedScheduling  0/1 nodes are available:
         pod has unbound immediate PersistentVolumeClaims.
```

워크샵은 `alertmanager.persistentVolume.enabled: false`로 persistence를 끄려고 한다.
지금 chart는 그 key가 `alertmanager.persistence.enabled`로 바뀌어 있다.
옛 key는 조용히 무시되고 StatefulSet은 여전히 PVC를 요구한다.

여기에 하나가 더 겹쳤다. 이 cluster에는 default StorageClass가 없다.

```
NAME   PROVISIONER             RECLAIMPOLICY   VOLUMEBINDINGMODE
gp2    kubernetes.io/aws-ebs   Delete          WaitForFirstConsumer
```

`gp2`는 있는데 `(default)` 표시가 없다. storageClassName을 안 준 PVC는 bind될 곳을 못 찾는다.
pod은 PVC를 기다리고 PVC는 StorageClass를 기다린다.

`alertmanager.persistence.enabled: false`를 넣고 `helm upgrade`를 돌렸다. 거부당했다.

```
Error: UPGRADE FAILED: cannot patch "prometheus-alertmanager" with kind StatefulSet:
StatefulSet.apps "prometheus-alertmanager" is invalid: spec: Forbidden:
updates to statefulset spec for fields other than 'replicas', 'ordinals', 'template',
'updateStrategy', 'revisionHistoryLimit', 'persistentVolumeClaimRetentionPolicy'
and 'minReadySeconds' are forbidden
```

StatefulSet의 volume claim template은 수정이 안 된다.
`helm uninstall` 하고 PVC까지 지운 다음 다시 깔았다.

### 5.3 고치고 나서

둘 다 잡으니 전부 Running이 됐다.

```
NAME                                 READY   STATUS
grafana-6bc6df764c-hsnzx             1/1     Running
prometheus-alertmanager-0            1/1     Running
prometheus-kube-state-metrics-...    1/1     Running
prometheus-prometheus-node-exporter  1/1     Running
prometheus-prometheus-pushgateway    1/1     Running
prometheus-server-7f57d49c54-shh8g   2/2     Running
```

vLLM scrape target도 붙었다.

```
vllm-metrics  up  http://vllm-service.default.svc.cluster.local:8080/metrics
```

Grafana에는 dashboard 세 개가 provisioning됐다.
`kubernetes-cluster.json`, `kubernetes-pods.json`, 그리고 워크샵이 주는 `vllm-dashboard.json`이다.
파일이 pod 안에 제대로 들어갔는지와 provisioning 로그까지만 확인했고
UI로 직접 열어보지는 않았다. 보려면 port-forward하고 admin / vllm-admin-2024로 들어가면 된다.

```bash
kubectl port-forward -n monitoring svc/grafana 3000:80
```

### 5.4 vLLM이 내보내는 metric

Prometheus에 올라온 목록을 훑었다.

```
vllm:num_requests_running          vllm:num_requests_waiting
vllm:prompt_tokens_total           vllm:generation_tokens_total
vllm:time_to_first_token_seconds   vllm:time_per_output_token_seconds
vllm:request_prefill_time_seconds  vllm:request_decode_time_seconds
vllm:e2e_request_latency_seconds   vllm:request_inference_time_seconds
vllm:num_preemptions_total         vllm:gpu_cache_usage_perc
vllm:iteration_tokens_total        vllm:cache_config_info
```

함정이 하나 있다. 워크샵 dashboard의 "KV Cache Usage" panel이 쓰는
`vllm:gpu_cache_usage_perc`가 Neuron backend에서는 계속 0으로 나온다.
metric 이름에 `gpu`가 박혀 있는 데서 알 수 있듯 CUDA backend 전용 계측이다.
Neuron에서 KV cache 압박을 보려면 `vllm:num_requests_waiting`과
`vllm:num_preemptions_total`을 봐야 한다.

---

## 6. Lab 5 — 성능 측정

### 6.1 concurrency를 바꿔가며

요청 40개를 concurrency만 바꿔가며 돌렸다. prompt는 짧고 `max_tokens=100`이다.

| Concurrency | 전체 소요 | Throughput (req/s) | 평균 latency | p95 latency | 집계 단어/초 |
| --- | --- | --- | --- | --- | --- |
| 1 | 24.12s | 1.66 | 0.60s | 0.84s | 79.8 |
| 2 | 13.32s | 3.00 | 0.65s | 0.97s | 141.2 |
| 4 | 7.01s | 5.70 | 0.68s | 1.14s | 244.6 |
| 8 | 7.33s | 5.46 | 1.36s | 1.99s | 248.1 |
| 16 | 6.42s | 6.23 | 2.04s | 3.43s | 243.5 |

concurrency 4에서 throughput이 멈춘다. 245 단어/초 근처에서 더 안 올라간다.
앞에서 본 `MAX_NUM_SEQS`가 4였다.
vLLM이 한 batch에 sequence 4개까지만 넣으니 다섯 번째부터는 queue에서 기다린다.

1에서 4로 갈 때는 throughput이 1.66에서 5.70으로 3.4배 올랐다.
그런데 평균 latency는 0.60초에서 0.68초로 거의 그대로다.
continuous batching이 일하고 있다는 뜻으로 읽었다.
batch에 자리가 있는 동안은 요청을 더 넣어도 거의 공짜다.

4를 넘기면 뒤집힌다. throughput은 고정되고 latency만 늘어난다.
concurrency 16에서 평균 latency가 2.04초로 세 배가 됐는데 throughput은 제자리다.

`max_num_seqs`는 throughput과 latency 사이의 다이얼이다.
이 값보다 큰 concurrency를 밀어 넣으면 tail latency만 나빠진다.

왜 이렇게 되는지는 9.2절의 칩 쪽 지표에서 드러난다.

### 6.2 llmperf

`ray-project/llmperf`로 token 단위 지표를 뽑았다.
입력 평균 256 token, 출력 평균 100 token, concurrency 5, 요청 50개.

| 지표 | 값 |
| --- | --- |
| TTFT 평균 | 251 ms |
| TTFT 최소 / 최대 | 48 ms / 621 ms |
| Inter-token latency 평균 | 12.03 ms |
| End-to-end latency 평균 | 1.175 s |
| 요청당 output throughput | 84.7 tok/s |
| 전체 output throughput | 336.6 tok/s |
| 분당 완료 요청 | 205.8 |
| 실패 요청 | 0 |

inter-token latency 12ms면 stream 하나당 83 tok/s쯤이다.
concurrency 5에서 전체가 336.6 tok/s니까 stream당 84.7이 거의 그대로 곱해졌다.
concurrency 5는 `max_num_seqs=4`를 살짝 넘긴 지점이라
아직 queue 지연이 본격화되기 전이어서 이렇게 깔끔하게 나온 것 같다.

### 6.3 server 쪽 숫자

client 측정과 별개로 Prometheus에서 vLLM 자신의 계측도 뽑았다.
Lab 5 전체 트래픽 누적 기준이다.

| 지표 | 값 |
| --- | --- |
| 누적 성공 요청 | 291 |
| 누적 prompt token | 16,896 |
| 누적 generation token | 21,253 |
| 평균 end-to-end latency | 1.029 s |
| 평균 TTFT | 366 ms |
| 평균 TPOT | 9.21 ms |
| 평균 prefill 시간 | 48.96 ms |
| 평균 decode 시간 | 663.5 ms |
| Preemption 횟수 | 0 |

요청 하나가 1.029초 걸리는데 prefill이 49ms고 decode가 663ms다.

숫자로 바꿔보면 더 선명하다.
입력 260 token을 49ms에 처리했으니 prefill은 약 5,300 tok/s.
decode는 token당 9.21ms니까 108 tok/s. 50배 차이다.

prefill은 병렬화가 되는 행렬 연산이고 decode는 token을 하나씩 만드는 순차 연산이다.
batching이 decode에서 특히 중요한 이유가 여기 있고,
그래서 `max_num_seqs`가 전체 throughput을 좌우한다.
앞의 concurrency 실험과 이 표가 같은 얘기를 다른 각도에서 하고 있다.

preemption 0회는 KV cache가 부족해서 요청을 되돌린 적이 없다는 뜻이다.
`max_model_len=1024`에 sequence 4개면 메모리에 여유가 충분했던 모양이다.

---
## 7. Lab 6 — HPA를 붙였더니

CPU 사용률 70%를 기준으로 replica를 1개에서 3개 사이로 움직이는 HPA를 건다.

```yaml
metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
minReplicas: 1
maxReplicas: 3
```

설치 직후는 평화롭다.

```
NAME       REFERENCE                    TARGETS       MINPODS  MAXPODS  REPLICAS
vllm-hpa   Deployment/vllm-deployment   cpu: 0%/70%   1        3        1
```

### 7.1 부하를 걸면

워크샵 지시대로 pod 안에서 CPU를 8 core 분량 태운다.

```bash
kubectl exec $POD -- bash -c 'for i in {1..8}; do (while true; do :; done) & done; sleep 300'
```

HPA가 바로 반응한다.

```
Normal  SuccessfulRescale  New size: 2; reason: cpu resource utilization above target
Normal  SuccessfulRescale  New size: 3; reason: cpu resource utilization above target
```

```
NAME       REFERENCE                    TARGETS         MINPODS  MAXPODS  REPLICAS
vllm-hpa   Deployment/vllm-deployment   cpu: 196%/70%   1        3        3
```

잘 되는 것처럼 보인다. pod 목록을 보기 전까지는.

```
NAME                              READY   STATUS    AGE
vllm-deployment-db754d957-9j6kp   0/1     Pending   2m17s
vllm-deployment-db754d957-lzvb5   1/1     Running   20m
vllm-deployment-db754d957-m56qf   0/1     Pending   3m17s
```

늘어난 두 개가 Pending에서 움직이지 않는다.

```
Warning  FailedScheduling  my-scheduler
  0/1 nodes are available:
    1 Insufficient aws.amazon.com/neuron,
    1 Insufficient cpu,
    1 Insufficient ephemeral-storage.
  preemption: 0/1 nodes are available: 1 No preemption victims found for incoming pod.
```

세 가지가 한꺼번에 모자란다. node는 한 대고 그 node의 Neuron device 1개는
이미 실행 중인 pod이 잡고 있다. CPU도 8000m 요청 중 하나가 나갔고
ephemeral-storage도 50Gi씩 요구한다. 어느 쪽으로 봐도 자리가 없다.

### 7.2 그래서 이게 무슨 소용인가

여기서 CPU를 196%까지 올린 건 추론이 아니라 busy loop다.
실제 추론 부하에서는 어떤지 재봤다.
concurrency 16으로 요청 300개를 계속 밀어 넣어 server를 완전히 포화시킨 상태다.

| 측정 대상 | 값 |
| --- | --- |
| 평균 latency | 2.95 s |
| p95 latency | 3.70 s |
| Throughput | 5.28 req/s (천장) |
| Pod CPU | 319~384 m (request 4000m 대비) |
| HPA가 읽은 값 | cpu: 9%/70% |
| Node CPU | 4% |

latency가 3초까지 늘어난 순간에도 HPA는 9%를 보고 있었다. threshold 70%에는 근처도 못 간다.
연산은 Neuron 칩에서 일어나고 host CPU는 tokenizing과 scheduling 정도만 한다.

결국 이 HPA를 발동시키려면 busy loop를 돌려야 한다.

scale 대상도 문제다. 가속기 1개당 pod 1개면 pod을 늘리려면 node를 늘려야 하는데
이 랩에는 Cluster Autoscaler도 Karpenter도 없다.
새 node가 뜬다 해도 Neuron compile과 모델 로딩에 몇 분이 걸린다.
초 단위로 반응해야 하는 트래픽에는 이 경로가 맞지 않는다.

쓸 만한 신호는 vLLM이 이미 내보내고 있다.
`vllm:num_requests_waiting`이 0보다 크면 이미 포화고,
`vllm:num_requests_running`이 `max_num_seqs`에 가까워지면 곧 포화다.
`vllm:num_preemptions_total`은 KV cache 압박을 알려주고,
`vllm:time_to_first_token_seconds`는 사용자 체감에 가장 가깝다.
`prometheus-adapter`로 custom metrics API에 노출하면 HPA가 이 지표로 동작한다.
10.3절에서 실제로 붙여봤다.

### 7.3 scale down은 정상이었다

부하가 걷히자 HPA는 한 번에 1개로 되돌렸다.

```
Normal  SuccessfulRescale  5m35s  New size: 2; reason: cpu resource utilization above target
Normal  SuccessfulRescale  4m35s  New size: 3; reason: cpu resource utilization above target
Normal  SuccessfulRescale    19s  New size: 1; reason: All metrics below target
```

`scaleDown` 정책이 60초당 50%인데 3에서 바로 1로 떨어졌다.
Pending으로 한 번도 Ready가 못 된 pod은 정책 계산에서 빠지는 것 같은데
문서로 확인하지는 않았다.

### 7.4 CPU 경합이 추론에 미치는 영향

부하 생성기가 node CPU를 101%까지 밀어 올린 상태에서
같은 추론 부하를 다시 돌려봤다. concurrency 4, 요청 20개다.

| | 정상 상태 | CPU 포화 상태 | 변화 |
| --- | --- | --- | --- |
| Throughput | 5.70 req/s | 3.74 req/s | 34% 감소 |
| 평균 latency | 0.68 s | 1.03 s | 51% 증가 |
| p95 latency | 1.14 s | 1.31 s | 15% 증가 |
| 실패율 | 0% | 0% | 변화 없음 |

연산은 Neuron이 하지만 tokenizing, scheduling, HTTP 처리, tensor 전송 준비는
host CPU 몫이다. CPU가 모자라면 가속기가 놀아도 throughput이 떨어진다.
가속기 node에서 CPU request를 인색하게 잡으면 손해다.

---

## 8. 워크샵 문서에서 걸린 것들

돌리면서 마주친 문제를 모았다. 같은 워크샵을 할 사람에게는 이게 제일 쓸모 있을 것 같다.

| # | 위치 | 증상 | 원인 | 해결 |
| --- | --- | --- | --- | --- |
| 1 | Lab 4 | `prometheus-server` `CrashLoopBackOff` | values의 scrape job 이름 `kubernetes-pods`가 chart 기본 job과 충돌 | job 이름을 `annotated-pods` 등으로 변경 |
| 2 | Lab 4 | `prometheus-alertmanager-0` 영구 `Pending` | `alertmanager.persistentVolume.enabled`는 폐기된 key. 현재는 `alertmanager.persistence.enabled`. cluster에 default StorageClass도 없음 | `alertmanager.persistence.enabled: false` 추가 후 재설치 |
| 3 | 사이드바 | `/workshop/vllm/ingress` 링크가 Page not found | 콘텐츠 누락 | 무시. 실제 Ingress 랩은 `ingress-k8s` |
| 4 | Lab 3·4·5 | `export CLUSTER_NAME=vllm-trn1-eks-cluster` | 실제 cluster 이름과 불일치 | 대부분 무해. Lab 4의 CloudWatch dashboard만 잘못된 이름으로 생성됨 |
| 5 | architecture | worker AMI를 `ami-08695d32a8bb6c5a5`로 명시 | 문서가 뒤처짐 | SSM 조회값 사용 |
| 6 | Lab 1 | node storage를 500GB로 설명 | manifest는 `volumeSize: 100` | 문서 오류 |
| 7 | Lab 2 | "재실행 시 20초면 뜬다" | S3 cache에 weights는 없어서 매번 다시 받음. 실측 41초 | 기대치 조정 |
| 8 | Lab 2 | `S3_PREFIX: compiled-models` | 실제 경로는 `cache/`. 이 변수는 안 쓰이는 듯 | 무해하지만 헷갈림 |
| 9 | teardown | AWS Load Balancer Controller 제거 절차 포함 | 이 워크샵은 그걸 설치하지 않음 | 무시 |
| 10 | Lab 4 dashboard | "KV Cache Usage" panel이 항상 0 | CUDA backend 전용 metric으로 보임 | queue와 preemption 지표로 대체 |
| 11 | Lab 2 | pod이 서빙 못 하는데 Ready로 표시됨 | readiness·liveness·startup probe가 전부 없음. 실측 29초 차이 | vLLM `/health`로 readinessProbe 추가. 배포 중 실패율이 43.8%에서 23.2%로 떨어졌다 (10.1절) |
| 12 | Lab 2 | 큰 모델을 올리면 노드 전체가 다운됨 | 컨테이너에 memory request·limit이 전혀 없음 | limit을 걸면 OOM이 컨테이너에서 끝난다 (12장) |
| 13 | Lab 2 | 기본 배포 전략으로는 rollout이 끝나지 않음 | maxSurge 1이면 새 pod이 가속기를 못 잡아 영원히 Pending | maxSurge 0으로 바꾸거나 노드를 하나 더 확보 |
| 14 | Lab 4 | Alertmanager에 규칙이 하나도 없음 | 설치만 되고 알림 규칙 미제공 | queue·TTFT·Pending 기준 규칙 추가 (10.4절) |

1번과 2번은 해당 컴포넌트가 아예 기동하지 않는다.
워크샵 콘텐츠는 정적인데 Helm chart의 values key는 그와 무관하게 바뀐다.

---

## 9. 랩을 끝내고 더 재본 것

앞 장까지는 클라이언트에서 본 숫자다. 칩 쪽에서는 어떻게 보이는지 세 가지를 더 쟀다.
`neuron-monitor`가 vLLM container 안에 이미 들어 있다.

### 9.1 NeuronCore는 concurrency 1에서도 이미 바쁘다

concurrency를 바꿔가며 `neuron-monitor`로 core 사용률을 샘플링했다.

| Concurrency | NeuronCore 0 | NeuronCore 1 | Throughput | 평균 latency | device memory |
| --- | --- | --- | --- | --- | --- |
| 0 (idle) | 0.0% | 0.0% | | | 3.63 GiB |
| 1 | 80.8% | 80.8% | 1.55 req/s | 0.64s | 3.63 GiB |
| 4 | 83.0% | 82.9% | 5.18 req/s | 0.77s | 3.63 GiB |
| 16 | 83.1% | 83.1% | 5.28 req/s | 2.96s | 3.63 GiB |

요청 하나만 보내도 core가 81%다. 거기서 throughput이 3.4배 올라가는 동안
사용률은 81에서 83으로 2포인트 움직였다.

그러니까 이 지표는 core가 뭔가를 실행 중인 시간의 비율이지 여유 용량이 아니다.
continuous batching이 decode step을 쉴 틈 없이 이어 붙이니까
sequence가 하나든 넷이든 core는 계속 바쁘다.

HPA 얘기와 이어진다. CPU가 못 쓸 신호라는 건 앞에서 봤는데,
NeuronCore 사용률도 마찬가지로 못 쓴다. 한가할 때 81%, 포화일 때 83%다.
결국 쓸 만한 건 queue 길이뿐이다.

device memory가 3.63 GiB에서 전혀 안 움직이는 것도 눈에 띈다.
`max_model_len=1024`에 `max_num_seqs=4`면 KV cache가 미리 잡히고 그게 끝이다.
칩에는 32 GB가 있다. 11%만 쓰고 있다.

### 9.2 decode step 시간은 batch size와 무관하다

`neuron-monitor`는 NEFF 실행 지연도 준다.
on-device 실행 p50이 concurrency 1에서 6.53 ms, 4에서 6.53 ms, 16에서 6.54 ms였다.
p99는 각각 35.0 / 40.3 / 40.3 ms다.

p50이 소수점 둘째 자리까지 같다. decode step 하나가 칩에서 6.53 ms 걸리고,
batch에 sequence를 넷 넣든 하나 넣든 그 시간은 그대로다.
p99가 35~40 ms인 건 prefill graph 실행이 섞여 들어간 것으로 보인다.

여기서 앞의 숫자들이 맞물린다.
step 하나에 6.53 ms인데 batch에 4개가 들어 있으면 그 시간에 token 4개가 나온다.
batch가 1이면 1개다. throughput이 batch size에 거의 비례하고
latency는 거의 안 변하는 이유가 이거다.

vLLM이 보고한 TPOT은 9.21 ms였다. 칩에서 6.53 ms를 쓰고
나머지 2.7 ms가 host 쪽에서 나간다. 대략 29%다.
sampling, detokenize, scheduler 같은 것들일 텐데 정확히 어디인지는 나눠보지 않았다.

### 9.3 prefill이 입력 길이와 무관했다

입력 길이만 바꿔가며 `max_tokens=1`로 재봤다.

| prompt tokens | p50 latency | token당 |
| --- | --- | --- |
| 18 | 47.7 ms | 2.652 ms |
| 66 | 47.5 ms | 0.720 ms |
| 130 | 47.6 ms | 0.366 ms |
| 258 | 48.0 ms | 0.186 ms |
| 386 | 47.1 ms | 0.122 ms |
| 450 | 47.2 ms | 0.105 ms |
| 602 | 48.4 ms | 0.080 ms |
| 702 | 48.6 ms | 0.069 ms |
| 802 | 48.8 ms | 0.061 ms |

입력이 18 token이든 802 token이든 47~49 ms다. 44배 차이인데 latency는 3% 움직인다.

`enable_bucketing: false` 때문이다.
bucketing을 끄면 NxD가 `max_model_len`에 맞춰 graph를 하나만 compile하고,
모든 요청이 그 1024짜리 graph를 padding해서 돈다.
18 token 프롬프트도 1024 token어치 연산을 한다.

token당 비용이 2.652 ms에서 0.061 ms로 43배 차이가 난다.
짧은 프롬프트가 압도적으로 손해다. 챗봇처럼 짧은 입력이 많은 워크로드라면
bucketing을 켜야 하는 이유가 여기 있다.

앞 장에서 vLLM metric으로 잰 평균 prefill 시간이 48.96 ms였다.
지금 잰 47~49 ms와 맞는다. 서로 다른 경로로 잰 값이 일치하니 믿어도 될 것 같다.

### 9.4 S3 cache가 있으면 41초에 뜬다

워크샵은 cache가 있으면 20초면 뜬다고 한다. pod을 지우고 다시 재봤다.

| 시각 | 사건 |
| --- | --- |
| 09:58:41 | pod 생성 |
| 09:58:43 | init container 시작하고 1초 안에 종료 |
| 09:58:45 | vLLM server container 시작 |
| 09:58:53 | Kubernetes가 pod을 Ready로 표시 |
| 09:59:22 | `Application startup complete` |

init container 로그는 이렇게 찍혔다.

```
Model cache exists, skipping compilation
```

compile은 확실히 건너뛰었다. 3분 32초가 1초 미만으로 줄었다.
그런데 실제로 요청을 받기 시작한 건 41초 뒤다. cold start 4분 9초와 비교하면 크게 줄었지만
워크샵이 말한 20초는 아니다.

41초가 어디로 갔는지 보면, container 안 HF cache가 4.2 GB로 차 있었다.
새 container니까 weights를 다시 받아온 것이다.
로그상 weight sharding과 dtype 변환에 13초가 들었고 나머지는 import와 engine 초기화다.

### 9.5 이 김에 발견한 것: readiness probe가 없다

위 표에서 Kubernetes는 09:58:53에 pod을 Ready로 표시했다.
그런데 server가 요청을 받기 시작한 건 09:59:22다. 29초 차이다.

manifest를 확인했다.

```json
{ "readinessProbe": null, "livenessProbe": null, "startupProbe": null }
```

probe가 하나도 없다. container process가 뜨는 순간 Ready가 되고
Service Endpoints에 들어간다. 그 29초 동안 들어온 요청은 연결이 거부된다.

랩에서는 replica가 하나뿐이고 부하도 없으니 티가 안 난다.
rolling update를 하거나 HPA가 실제로 pod을 늘리는 순간 바로 드러날 문제다.
vLLM은 `/health`를 이미 제공하니 probe를 붙이는 건 몇 줄이면 된다.

---

## 10. 운영이라고 생각하고 다시 해본 것

랩은 "돌아간다"까지만 본다. 운영은 배포하고, 장애 나고, 과부하 걸리고,
새벽에 알림 받는 쪽이다. 클러스터가 아직 살아 있길래 네 가지를 더 해봤다.

### 10.1 무중단 배포가 되긴 하는가

probe가 없다는 건 9.5절에서 알았는데 영향은 재지 않았다.
초당 5건씩 요청을 보내면서 `kubectl rollout restart`를 돌리고 실패를 셌다.
세 가지 상태로 각각 재봤다.

| 시나리오 | 배포 전략 | probe | 실패율 | 장애 구간 | rollout 완료 |
| --- | --- | --- | --- | --- | --- |
| A | maxSurge 1 / maxUnavailable 0 (기본) | 없음 | 0% | 없음 | 안 됨 |
| B | maxSurge 0 / maxUnavailable 1 | 없음 | 43.8% | 95.2s | 됨 |
| C | maxSurge 0 / maxUnavailable 1 | 있음 | 23.2% | 64.7s | 됨 |

시나리오 A에서는 748건을 보내 하나도 실패하지 않았는데 배포가 끝나지 않았다.

```
vllm-deployment-85d9454bdb-x9rx8   0/1   Pending   2m8s
vllm-deployment-db754d957-thrlc    1/1   Running   117m
```

```
Waiting for deployment "vllm-deployment" rollout to finish: 1 old replicas are pending termination...
```

기본 전략은 새 pod이 Ready가 된 다음 기존 pod을 내린다.
그런데 Neuron device는 하나뿐이고 기존 pod이 잡고 있다. HPA 때 본 그 에러가 그대로 난다.

```
0/1 nodes are available: 1 Insufficient aws.amazon.com/neuron, 1 Insufficient cpu,
1 Insufficient ephemeral-storage.
```

서비스는 멀쩡한데 배포만 조용히 멈춘다. CI에서 `rollout status`를 기다리고 있었다면 거기서 끝난다.
그리고 이 상태에서 누가 답답해서 `kubectl delete pod`을 치면 그때 장애가 난다.

`maxSurge: 0`으로 바꾸니 배포는 진행된다. 대신 기존 pod을 먼저 죽인다.
95초 동안 요청의 44%가 실패했다. 실패는 전부 connection refused다.
그중 앞 49초는 Endpoints가 비어 있던 구간이고, 뒤 25초는 새 pod이 Ready로 표시됐는데
아직 서빙을 못 하던 구간이다. 뒤쪽 25초가 probe가 없어서 생긴 몫이다.

probe를 붙였다. vLLM이 `/health`를 이미 제공하니 몇 줄이면 된다.

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

같은 걸 다시 돌리니 실패가 23.2%로 줄고 장애 구간이 64.7초가 됐다.
Ready로 표시된 시각이 `Application startup complete`보다 2초 뒤다.
전에는 29초 앞섰으니 방향이 뒤집혔다.

남은 64.7초는 probe로 못 줄인다. 가속기가 하나라 기존 pod이 죽어야 새 pod이 뜬다.
무중단으로 하려면 노드가 하나 더 있어야 하는데, 이 계정의 Trn quota는 8 vCPU고
`trn1.2xlarge`가 정확히 8 vCPU다. 두 대를 못 띄운다.
가속기 워크로드에서 무중단 배포는 N+1 용량을 사는 문제지 설정 문제가 아니다.

### 10.2 과부하에서 어떻게 깨지는지 보려 했는데 안 깨졌다

concurrency를 올려가며 어디서 5xx가 나는지 보려고 했다.

| Concurrency | 전체 소요 | Throughput | p50 | p95 | p99 | 실패 |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 18.5s | 10.81 req/s | 0.59s | 1.43s | 1.67s | 0 |
| 32 | 18.7s | 10.71 req/s | 2.83s | 3.63s | 3.76s | 0 |
| 64 | 18.7s | 10.69 req/s | 5.69s | 6.61s | 6.94s | 0 |
| 128 | 18.3s | 10.93 req/s | 9.31s | 12.08s | 12.50s | 0 |
| 256 | 18.3s | 10.94 req/s | 9.09s | 17.35s | 17.84s | 0 |

256까지 올려도 에러가 0이다. throughput은 10.8 req/s에 고정되고 latency만 늘어난다.
p99가 1.67초에서 17.84초가 됐다.

vLLM에는 admission control이 없다. 들어오는 요청을 전부 받아서 queue에 쌓는다.
클라이언트 쪽에 backpressure 신호가 가지 않는다는 뜻이다.
타임아웃 5초짜리 클라이언트라면 진작 포기했을 요청을 서버는 계속 계산하고 있다.
그 계산은 아무도 안 기다린다.

앞에 뭔가를 둬야 한다. ingress에서 `limit_conn`이나 `limit_req`로 동시 연결을 막든,
gateway에서 queue 깊이를 보고 429를 돌려주든. 서버가 알아서 거절해주길 기대하면 안 된다.

### 10.3 queue 기반 HPA를 실제로 붙여봤다

7장에서 CPU 기준 HPA가 쓸모없다는 건 봤다. 대안으로 queue 길이를 말만 했지 해보지는 않았다.
prometheus-adapter로 external metric을 만들었다.

```yaml
rules:
  default: false
  external:
    - seriesQuery: '{__name__="vllm:num_requests_waiting"}'
      resources: { namespaced: false }
      name: { as: "vllm_requests_waiting" }
      metricsQuery: "max(<<.Series>>)"
```

HPA는 이걸 본다.

```yaml
metrics:
  - type: External
    external:
      metric: { name: vllm_requests_waiting }
      target: { type: Value, value: "2" }
```

concurrency 32로 부하를 걸었다.

```
[12:12:58] waiting=0   running=0  hpa=0/2   replicas=1
[12:13:23] waiting=28  running=4  hpa=28/2  replicas=1  (Pending pod 1개 생성)
[12:13:47] waiting=28  running=4  hpa=28/2  replicas=2
[12:14:11] waiting=28  running=3  hpa=28/2  replicas=3
[12:17:02] waiting=0   running=0  hpa=0/2   replicas=3
```

25초 만에 반응했다. `running`이 4에서 멈춰 있는 게 `max_num_seqs=4`고
나머지 28개가 `waiting`이다. 지표가 정확히 상태를 말해준다.
같은 부하에서 CPU 기준 HPA는 9%를 보고 있었다.

늘어난 pod은 여전히 Pending이다. 신호를 고쳤다고 용량이 생기지는 않는다.
실제로 쓰려면 Karpenter로 Neuron node pool을 붙여서
HPA가 pod을 늘리면 노드가 따라 붙게 해야 한다.

### 10.4 error rate로 알림을 걸면 아무것도 안 울린다

Alertmanager는 랩에서 설치만 되고 규칙이 하나도 없다. 네 개를 넣었다.

| 규칙 | 조건 | for |
| --- | --- | --- |
| VLLMQueueBacklog | `vllm:num_requests_waiting > 2` | 1m |
| VLLMHighTTFT | TTFT p95 > 1s | 2m |
| VLLMPodPending | vLLM pod이 Pending | 2m |
| VLLMNoReadyPod | Ready인 pod이 0개 | 1m |

concurrency 32로 한 번 부하를 걸었더니 세 개가 순서대로 올라왔다.

```
[12:22:28] (없음)
[12:22:48] QueueBacklog:pending  HighTTFT:pending  PodPending:pending
[12:23:48] QueueBacklog:firing   HighTTFT:pending  PodPending:pending
[12:24:48] QueueBacklog:firing   HighTTFT:firing   PodPending:firing
```

그 5분 동안 요청 1600건이 전부 성공했다. 실패 0건이다.
error rate 기준으로 알림을 걸어놨다면 아무 일도 없었던 것처럼 지나간다.
사용자는 6초씩 기다리고 있는데 대시보드는 초록색이다.

`VLLMPodPending`은 10.1의 멈춘 배포도 잡는다.
그 상황도 요청은 100% 성공하고 있었으니 다른 지표로는 안 보인다.

### 10.5 토큰당 얼마인가

`trn1.2xlarge` 온디맨드가 시간당 1.34달러다. 초당 0.000372달러다.

| 조건 | output throughput | 1M token당 |
| --- | --- | --- |
| concurrency 1 | 84.7 tok/s | 4.39 달러 |
| concurrency 5 | 336.6 tok/s | 1.11 달러 |

같은 하드웨어에서 4배 차이가 난다. batching을 얼마나 채우느냐가 그대로 단가다.
`max_num_seqs=4`라는 설정 한 줄이 비용 4배를 좌우한다.
9.1절에서 본 대로 device memory는 32 GB 중 3.63 GB밖에 안 쓰고 있으니
올릴 여지가 있다. 11.5절에서 실제로 올려봤다.

이 숫자는 1.1B 모델 기준이다. 큰 모델은 token당 연산이 늘어나니 단가도 따라 오른다.
그리고 여기엔 EKS 컨트롤 플레인과 로드밸런서 비용이 안 들어가 있다.

---

## 11. 용량과 비용을 건드려봤다

10장을 끝내고 두 가지가 걸렸다.
하나는 Pending으로 쌓인 pod을 실제로 해소해보는 것이고,
다른 하나는 `max_num_seqs`를 올리면 비용이 어디까지 떨어지는지였다.

### 11.1 같은 인스턴스로는 용량을 못 늘린다

10.1절에서 무중단 배포가 결국 용량 문제라고 정리했다. 그럼 노드를 하나 더 붙이면 된다.
노드그룹을 2대로 올렸더니 이렇게 막혔다.

```
Could not launch On-Demand Instances. VcpuLimitExceeded - You have requested more vCPU
capacity than your current vCPU limit of 8 allows for the instance bucket that the
specified instance type belongs to.
```

`trn1.2xlarge`가 정확히 8 vCPU다. Trn 계열 한도가 8이니 한 대가 천장이다.
가속기 워크로드의 용량 계획은 계열별 vCPU 한도로 세워야 한다. 인스턴스 대수가 기준이 아니다.
오토스케일러를 붙여도 이 벽은 그대로다.

### 11.2 다른 계열로 용량을 채운다

Inf 한도는 Trn과 별도로 8 vCPU가 열려 있고 `inf2.xlarge`가 4 vCPU다.
기존 노드 role을 재사용해서 노드그룹을 하나 더 만들었다.

```yaml
managedNodeGroups:
  - name: neuron-inf2-1x
    instanceType: inf2.xlarge
    iam:
      instanceRoleARN: arn:aws:iam::...:role/eksctl-...-NodeInstanceRole-...
```

붙고 나서 두 노드를 비교해봤다.

| | trn1.2xlarge | inf2.xlarge |
| --- | --- | --- |
| Neuron device | 1 | 1 |
| NeuronCore | 2 | 2 |
| 가속기 메모리 | 32 GB | 32 GB |
| 호스트 vCPU | 7910m | 3920m |
| 호스트 메모리 | 31 GB | 15 GB |
| 시간당 | $1.34 | $0.76 |

가속기 쪽은 완전히 같다. 둘 다 NeuronCore-v2다. 호스트만 절반이다.
기존 deployment는 CPU를 4000m 요청해서 inf2에 안 들어간다. 요청을 2000m로 낮춘
별도 deployment를 만들어 올렸다.

### 11.3 compile cache가 두 칩에서 공유된다

가장 궁금했던 부분이다. Trainium용으로 구운 NEFF를 Inferentia2가 읽을까.

```
INFO:Neuron:Done Sharding weights in 29.42s
INFO [neuronx_distributed.py:244] Successfully loaded precompiled model artifacts from /shared/model/cache
INFO:     Application startup complete.
```

읽는다. 재compile 없이 그대로 떴다.
S3 cache 하나를 두 종류 노드가 같이 쓸 수 있다는 뜻이다.
용량이 모자랄 때 가속기 종류를 섞어서 채울 수 있다.

두 칩 모두 NeuronCore-v2 세대라 가능한 것이다.

### 11.4 두 칩의 추론 성능은 같았다

같은 모델, 같은 설정, 같은 부하로 나란히 재봤다.

| Concurrency | trn1.2xlarge | inf2.xlarge |
| --- | --- | --- |
| 1 | 1.54 req/s, 0.65s | 1.40 req/s, 0.71s |
| 4 | 4.92 req/s, 0.78s | 5.08 req/s, 0.77s |
| 16 | 5.68 req/s, 2.25s | 5.68 req/s, 2.28s |

노이즈 범위다. 같은 칩이니 당연하기도 하다.
그런데 가격은 trn1이 1.76배 비싸다.
Trainium은 학습용이라 inter-chip interconnect와 넉넉한 호스트 자원에 값이 붙어 있는데,
추론만 하면 그걸 안 쓴다.

추론 워크로드라면 inf2가 맞다. 이 워크샵이 trn1을 쓰는 건 Trainium 워크샵이기 때문이다.

### 11.5 max_num_seqs를 올리면 단가가 떨어진다

9.1절에서 device memory를 32 GB 중 3.63 GiB밖에 안 쓰는 걸 봤다.
값을 4에서 32까지 올려가며 재봤다. 값마다 S3 cache를 비우고 재compile했다.

| max_num_seqs | Throughput | output tok/s | 평균 latency | p95 | device memory | compile | $/1M (trn1) | $/1M (inf2) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 5.10 req/s | 248 | 3.00s | 3.78s | 3.63 GiB | 214s | 1.50 | 0.85 |
| 8 | 7.73 req/s | 377 | 3.99s | 4.77s | 3.72 GiB | 217s | 0.99 | 0.56 |
| 16 | 10.63 req/s | 518 | 5.80s | 6.88s | 3.89 GiB | 221s | 0.72 | 0.41 |
| 32 | 13.92 req/s | 678 | 8.89s | 10.48s | 4.24 GiB | 229s | 0.55 | 0.31 |

부하는 매번 `max_num_seqs`의 4배 concurrency로 걸었다.
token 수치는 `vllm:generation_tokens_total`을 Prometheus에서 읽어
요청당 48.7 token으로 환산한 것이다. 이 workload 기준이라 10.5절 숫자와는 기준이 다르다.

4에서 32로 가면 throughput이 2.73배가 되고 단가가 2.73배 떨어진다.
그리고 p95 latency가 2.77배 나빠진다. 거의 1대1 교환이다.
공짜가 없다. 지연을 팔아 비용을 산다.

몇 가지가 눈에 띈다.

device memory는 3.63에서 4.24 GiB로 0.6 GiB밖에 안 늘었다. 여전히 32 GB의 13%다.
batch를 8배로 키웠는데 KV cache가 이만큼밖에 안 는다.
`max_model_len`이 1024라 그렇다. 메모리는 한계가 아니었다.

compile 시간은 214초에서 229초로 거의 안 변한다. batch size는 compile 비용에 영향이 없다.

실패는 네 값 모두 0건이다. concurrency 128에서도 안 깨진다. 10.2절과 같은 얘기다.

### 11.6 두 레버를 합치면

워크샵 기본값은 `trn1.2xlarge`에 `max_num_seqs=4`다. 1M output token당 1.50달러다.
`inf2.xlarge`에 `max_num_seqs=32`면 0.31달러다. 4.8배 차이다.

하드웨어도 모델도 그대로다. 인스턴스 종류 하나와 설정값 하나를 바꿨을 뿐이다.
그 대신 p95 latency가 3.78초에서 10.48초가 된다.
SLO가 허락하는 만큼만 가져가면 된다.

---

## 12. 모델 크기의 상한은 호스트가 정한다

지금까지 잰 건 전부 TinyLlama-1.1B다. 모델을 키우면 무엇이 먼저 막히는지 보려고
8B를 올려봤다. 컨테이너 안의 NxD가 지원하는 아키텍처부터 확인했다.

```
$ ls /opt/conda/lib/python3.10/site-packages/neuronx_distributed_inference/models/
dbrx  deepseek  llama  llama4  mixtral  mllama  qwen2  qwen3  qwen3_moe
```

`qwen3`가 있어서 Qwen3-8B로 정했다.

### 12.1 8B를 올리면 노드가 통째로 내려간다

매니페스트를 그대로 쓰고 Qwen3-8B를 올리면 init container가 weights를 받아
그래프를 만드는 동안 호스트 메모리가 고갈된다. 6분 뒤 kubelet이 상태 보고를 멈춘다.

```
MemoryPressure   Unknown   NodeStatusUnknown   Kubelet stopped posting node status.
```

S3 CSI 소켓도 같이 사라진다.

```
MountVolume.SetUp failed for volume "s3-model-cache-pv" :
  dial unix /var/lib/kubelet/plugins/s3.csi.aws.com/csi.sock: connect: no such file or directory
```

pod 하나가 감당 못 할 모델을 집었을 뿐인데 노드 전체가 스케줄 불가 상태가 된다.
복구하려면 노드그룹을 0으로 줄였다가 다시 올려 인스턴스를 교체해야 한다.

### 12.2 매니페스트에 메모리 제한이 없다

노드가 죽은 이유는 매니페스트에 있다.

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
제한이 없으니 컨테이너가 호스트 메모리를 다 먹을 때까지 자라고,
그 끝에서 커널이 kubelet을 포함해 아무거나 죽인다.

### 12.3 제한을 걸면 컨테이너만 죽는다

Qwen3-4B로 낮추고 메모리 limit 24Gi를 걸어 올렸다.

```
exitCode: 137
reason: OOMKilled
startedAt:  15:57:49
finishedAt: 16:02:59
restartCount: 1
```

컨테이너만 죽고 쿠버네티스가 재시작했다. 노드는 Ready 그대로고
메모리도 1.98 GiB로 바로 돌아왔다. 같은 원인, 다른 결과다.

28Gi로 올려도 4분 27초 만에 OOMKilled였다.
노드의 allocatable memory가 31315416Ki, 약 29.9 GiB라 28Gi가 사실상 천장이다.

### 12.4 가속기 메모리와 호스트 메모리는 다른 제약이다

`trn1.2xlarge`의 Neuron device에는 32 GB가 있다. 4B든 8B든 bf16으로 들어간다.
그런데 그 weights를 칩에 올리려면 호스트가 먼저 들고 샤딩해야 한다.

Qwen3-4B 체크포인트는 12 GB이고 `torch_dtype`은 `bfloat16`이다.
그런데 로딩 시작 1분 만에 python 프로세스의 RSS가 8.6 GB가 되고 계속 자라 28 GB를 넘긴다.
체크포인트 크기의 세 배 가까이 쓴다.

로그에 단서가 있다.

```
UserWarning: Found torch.float32 weights in checkpoint: lm_head.weight. Will convert to torch.bfloat16
```

NxD가 로딩 과정에서 fp32를 거치고, tensor parallel 2로 나누면서 사본을 더 만든다.

| 모델 | 체크포인트 | 결과 |
| --- | --- | --- |
| TinyLlama-1.1B | 2.2 GB | 정상. device memory 3.63 GiB |
| Qwen3-4B | 12 GB | 호스트 28Gi 초과로 OOMKilled |
| Qwen3-8B | 16 GB | 제한이 없으면 노드 전체가 다운 |

이 인스턴스에서 실제로 올릴 수 있는 모델 상한은 1.1B와 4B 사이 어딘가다.
32 GB짜리 가속기를 달고도 그렇다. 병목은 호스트 31 GiB다.

더 큰 호스트를 살 수도 없었다. `inf2.xlarge`는 15 GiB로 더 작고,
`inf2.8xlarge`와 `trn1.32xlarge`는 이 계정의 quota를 넘는다.

큰 모델을 Neuron에 올리려면 가속기 개수만 보면 안 된다.
`trn1.32xlarge`처럼 호스트 메모리가 512 GB인 인스턴스를 고르거나,
체크포인트를 미리 샤딩해서 로딩 피크를 낮추는 경로를 찾아야 한다.
후자는 해보지 않았다.

---

## 13. 정리해두고 싶은 것

### 13.1 Trainium과 GPU의 차이

K8s에서 보이는 resource 이름부터 다르다.
NVIDIA는 `nvidia.com/gpu` 하나인데 Neuron은 `aws.amazon.com/neuron`과
`aws.amazon.com/neuroncore` 두 축이다.
`tensor_parallel_size`가 가리키는 것도 칩 안의 core 수다. GPU 장 수가 아니다.

가장 크게 다가온 건 AOT compile이다.
모델이나 설정을 바꾸면 재compile이 필요하고 compiler 버전이 바뀌어도 cache가 무효화된다.
배포 pipeline에 compile 단계를 명시적으로 넣고 cache를 artifact로 관리해야 한다.
이번 랩의 init container와 S3 조합이 그 최소 구현이다.

scheduling도 다르다. NeuronCore를 연속으로 잡아야 해서 scheduler extension이 필요하다.
vLLM engine도 V0로 내려간다. GPU에서는 신경 쓸 일이 없는 것들이다.

`enable_bucketing: false` 설정도 AOT 때문이다.
bucketing을 켜면 여러 입력 길이에 대해 graph를 미리 compile해서
runtime padding 낭비를 줄일 수 있지만 compile 시간과 cache 크기가 늘어난다.
랩은 시간을 아끼려고 꺼둔 설정이다.

### 13.2 측정에서 얻은 것

throughput 천장을 정한 건 하드웨어가 아니고 `max_num_seqs=4`였다.
benchmark할 때 이 값을 모르고 측정하면 Trainium이 느리다는 엉뚱한 결론으로 간다.
그 아래 기계는 단순했다. decode step 하나가 칩에서 6.53 ms 걸리고 그 시간은 batch size와 무관하다.
batch에 넷을 넣으면 같은 6.53 ms에 token이 넷 나온다.

prefill과 decode의 비대칭도 숫자로 처음 봤다.
입력 260 token prefill이 49ms, 출력 98 token decode가 663ms.
token당으로 환산하면 50배 차이다.
LLM serving 최적화가 왜 decode batching에 집착하는지 이해가 됐다.

그런데 그 prefill 49ms는 입력 길이와 상관이 없었다.
`enable_bucketing`을 끄면 모든 요청이 `max_model_len` 크기의 graph를 padding해서 돈다.
18 token 프롬프트가 802 token 프롬프트와 같은 시간을 쓴다.
설정 하나가 성능 특성을 통째로 바꿔놓는데 워크샵 문서에는 그 얘기가 없다.

용량 지표로 쓸 만한 게 거의 없다는 것도 알게 됐다.
CPU는 서버가 포화돼도 9%고, NeuronCore 사용률은 한가할 때 81% 포화일 때 83%다.
device memory는 KV cache가 미리 잡혀서 부하와 무관하게 고정이다.
움직이는 건 queue 길이와 latency뿐이다.

CPU도 가속기 workload에서 병목이 된다.
node CPU를 포화시켰더니 throughput이 34% 떨어졌다.

모델 크기의 상한은 가속기 메모리가 정하지 않았다. 호스트 메모리가 정했다.
32 GB짜리 칩을 달고도 4B를 못 올렸다. 로딩 경로가 체크포인트의 세 배를 쓴다.

비용을 움직이는 레버는 두 개였다. 인스턴스 종류와 `max_num_seqs`다.
추론만 하면 Trainium을 살 이유가 없고, batch를 채우면 단가가 그만큼 떨어진다.
둘을 합치면 4.8배인데 그 대가로 p95 latency가 2.8배가 된다.

### 13.3 실서비스라면 바꿀 것

10장에서 몇 개는 실제로 바꿔보고 효과를 쟀다. 정리하면 이렇다.

autoscaling 지표는 CPU에서 `vllm:num_requests_waiting`으로 바꿨다.
prometheus-adapter를 거쳐 external metric으로 올리면 HPA가 25초 만에 반응한다.
다만 노드가 따라 붙어야 의미가 있으니 Karpenter로 Neuron node pool을 붙여야 한다.

배포는 probe를 붙이고 `maxSurge`를 명시적으로 고른다.
가속기가 하나뿐이면 무중단은 포기하거나 N+1 용량을 사야 한다. 중간은 없다.
그 N+1을 같은 인스턴스로 채울 필요는 없다.
compile cache가 trn1과 inf2 사이에서 공유되니 가속기 종류를 섞어도 된다.

앞단에 과부하 방어가 필요하다. vLLM은 들어오는 요청을 전부 받아 queue에 쌓고
latency로만 티를 낸다. ingress나 gateway에서 동시 연결과 queue 깊이를 끊어야 한다.

알림은 error rate가 아니라 queue, TTFT, Pending pod을 봐야 한다.
1600건이 전부 성공하는 동안에도 사용자는 6초씩 기다릴 수 있다.

cold start는 weights와 NEFF를 모두 S3나 EFS에 두고 미리 예열한 pod으로 흡수한다.
외부 노출은 Service를 ClusterIP로 내리고 Ingress만 남긴다.
Prometheus도 in-memory 말고 AMP 같은 managed backend가 낫다.

컨테이너에 memory limit을 반드시 건다.
가속기 노드에서 호스트 메모리는 아무도 선언하지 않는 자원이고, 터지면 노드가 통째로 나간다.

secret 처리는 꼭 고쳐야 한다.
랩의 ConfigMap에는 `HF_TOKEN` 값이 평문으로 들어간다.
Secret을 따로 만들어 놓고도 ConfigMap에 같은 값을 또 넣는다.
ConfigMap은 RBAC 범위가 넓고 `kubectl describe`에도 그대로 나온다.

---
## 14. 랩 구간 타임라인

랩 여섯 개를 한 번에 쭉 돌렸을 때 각 단계가 얼마나 걸렸는지다. UTC 기준.
9장 이후의 추가 측정은 여기 안 들어 있고, 그쪽이 시간으로는 훨씬 길었다.

| 시각 | 단계 | 소요 |
| --- | --- | --- |
| 15:12 | AWS CLI, Helm, jq 설치 | 약 3분 |
| 15:14 | kubeconfig 설정, node용 SSH key 생성 | 30초 |
| 15:15 | subnet 탐색, node group 설정 작성 | 30초 |
| 15:15 - 15:17 | `eksctl create nodegroup` | 2분 41초 |
| 15:23 | S3 bucket 생성, Neuron device plugin 재설치 | 1분 |
| 15:24 | Neuron scheduler extension, S3 CSI driver | 1분 |
| 15:30 - 15:34 | 재배포. init compile 3분 32초 + server 기동 34초 | 4분 9초 |
| 15:35 | NGINX Ingress 설치 및 ELB 할당 | 49초 |
| 15:37 | Prometheus 1차 설치, job 이름 충돌로 실패 | 2분 |
| 15:38 | 재설치, Grafana dashboard mount | 3분 |
| 15:41 - 15:45 | load test, concurrency 실험, llmperf | 약 8분 |
| 15:45 - 15:53 | HPA 설치, scale up 시험, scale down 확인 | 약 8분 |

순수 실행 시간은 40분쯤이다. 워크샵이 제시한 예상 시간과 비슷하다.
단 8장의 결함을 고치는 시간은 여기 들어 있지 않다.

가장 오래 걸린 단일 작업은 첫 배포의 container image pull이었다.
Neuron vLLM image가 커서 3분 가까이 걸렸다.

---

## 15. 최종 상태

모든 측정을 마쳤을 때 cluster 구성이다. 노드가 두 대다.
`trn1.2xlarge`는 12장에서 한 번 죽어서 교체된 것이고, `inf2.xlarge`는 11장에서 추가한 것이다.

```
NAMESPACE       NAME                                    READY   STATUS
default         vllm-deployment-...                     1/1     Running   (trn1)
default         vllm-inf2-...                           1/1     Running   (inf2)
ingress-nginx   ingress-nginx-controller-...            1/1     Running
kube-system     k8s-neuron-scheduler-... / my-scheduler  1/1     Running
kube-system     neuron-device-plugin-...                1/1     Running   (노드마다)
kube-system     s3-csi-controller-... / s3-csi-node-...  1/1 3/3 Running
kube-system     metrics-server-...                      1/1     Running
monitoring      prometheus-server-...                   2/2     Running
monitoring      prometheus-adapter-...                  1/1     Running
monitoring      prometheus-alertmanager-0               1/1     Running
monitoring      grafana-...                             1/1     Running
```

Helm release는 neuron-helm-chart 1.10.0, aws-mountpoint-s3-csi-driver 2.8.0,
ingress-nginx 4.15.1, prometheus 29.28.1, grafana 10.5.15,
prometheus-adapter 5.3.0이다.

워크샵이 준 상태에서 바뀐 것들이다.

| 항목 | 워크샵 기본값 | 바꾼 값 | 근거 |
| --- | --- | --- | --- |
| probe | 없음 | `/health`로 startup·readiness·liveness | 10.1절 |
| 배포 전략 | maxSurge 1 / maxUnavailable 0 | maxSurge 0 / maxUnavailable 1 | 10.1절 |
| HPA 지표 | CPU 70% | `vllm_requests_waiting` > 2 | 10.3절 |
| 알림 규칙 | 없음 | queue·TTFT·Pending·NoReadyPod 네 개 | 10.4절 |
| `max_num_seqs` | 4 | 32 | 11.5절 |
| 노드 | trn1.2xlarge 한 대 | trn1 + inf2 두 대 | 11.2절 |
| memory limit | 없음 | 컨테이너마다 지정 | 12.2절 |

## 16. 마무리

`trn1.2xlarge` 한 대에서 시작해 서빙, 관측, 부하, 오토스케일링, 배포, 비용까지 봤다.
중간에 `inf2.xlarge`를 한 대 더 붙여 두 칩을 비교했다.

기억에 남는 숫자는 두 개다.

하나는 HPA가 읽은 9%다. 서버가 완전히 포화돼 latency가 3초까지 늘어난 순간에도
CPU 지표는 한 자릿수였다. CPU 기준 autoscaling은 가속기 workload에서 동작하지 않는다.

다른 하나는 prefill의 48 ms다. 입력이 18 token이든 802 token이든 같았다.
`enable_bucketing` 한 줄이 성능 특성을 통째로 바꿔놓는데 워크샵 문서에는 그 얘기가 없다.

설정값 하나가 하드웨어보다 크게 작용하는 장면을 여러 번 봤다.
throughput 천장을 정한 것은 `max_num_seqs`였고,
1M token 단가를 4.8배 바꾼 것도 인스턴스 종류와 그 값이었다.
그리고 모델 크기의 상한을 정한 것은 31 GiB짜리 호스트였다. 32 GB짜리 가속기가 아니었다.

다음에 볼 것을 세 가지 적어둔다.
`enable_bucketing`을 켰을 때 prefill이 얼마나 싸지고 compile 시간이 얼마나 늘어나는지,
`max_num_seqs`를 32보다 더 올리면 어디서 꺾이는지,
그리고 체크포인트를 미리 샤딩해두면 로딩 피크가 얼마나 내려가는지다.
마지막 것이 되면 작은 인스턴스에서도 큰 모델을 올릴 수 있다.

---

## 부록 A. 재현용 명령

전부 워크샵 EC2 인스턴스에서 실행한다.

```bash
# 환경
export AWS_REGION=us-west-2
export CLUSTER_NAME=<your-cluster>
export INSTANCE_TYPE=trn1.2xlarge
export WORKER_AMI=$(aws ssm get-parameter \
  --name /aws/service/eks/optimized-ami/1.33/amazon-linux-2023/x86_64/neuron/recommended/image_id \
  --region $AWS_REGION --query "Parameter.Value" --output text)
export BUCKET_NAME=<your-cache-bucket>

# cluster 접속
aws eks update-kubeconfig --region $AWS_REGION --name $CLUSTER_NAME

# trn1을 지원하는 AZ 확인
aws ec2 describe-instance-type-offerings --location-type availability-zone \
  --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
  --query 'InstanceTypeOfferings[*].Location' --output text

# node group
eksctl create nodegroup --config-file=eks_nodegroup.yaml

# Neuron 스택
helm upgrade --install neuron-helm-chart oci://public.ecr.aws/neuron/neuron-helm-chart \
  --set "scheduler.enabled=true" --set "npd.enabled=false"

# S3 CSI
helm repo add aws-mountpoint-s3-csi-driver https://awslabs.github.io/mountpoint-s3-csi-driver
helm upgrade --install aws-mountpoint-s3-csi-driver \
  --namespace kube-system aws-mountpoint-s3-csi-driver/aws-mountpoint-s3-csi-driver

# 이후는 저장소의 스크립트로
bash lab2-deploy-vllm.sh
bash lab3-ingress.sh
bash lab4-observability.sh
bash lab5-perftest.sh
bash lab6-hpa.sh
bash teardown.sh
```

Neuron resource가 제대로 보이는지 확인하는 한 줄.

```bash
kubectl get nodes -o custom-columns=\
NAME:.metadata.name,\
Neuron:.status.allocatable.aws\\.amazon\\.com/neuron,\
NeuronCore:.status.allocatable.aws\\.amazon\\.com/neuroncore
```

## 부록 B. 참고 자료

- [AWS Neuron 문서](https://awsdocs-neuron.readthedocs-hosted.com/)
- [vLLM User Guide for NxD Inference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide-v1.html)
- [vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron)
- [aws-neuron/aws-neuron-eks-samples](https://github.com/aws-neuron/aws-neuron-eks-samples)
- [ray-project/llmperf](https://github.com/ray-project/llmperf)
- [Mountpoint for Amazon S3 CSI Driver](https://github.com/awslabs/mountpoint-s3-csi-driver)
