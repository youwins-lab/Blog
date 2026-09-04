---
layout: post
title: "2.4T 파라미터 MoE 서빙: 8-GPU 한 노드 vs 16-GPU 두 노드"
date: 2026-09-01 10:00:00 +0900
categories: study llm-serving
---

<style>
.fig{margin:1.8em 0}
.fig svg{width:100%;height:auto;display:block;overflow:visible}
.fig figcaption{font-size:.82em;opacity:.62;margin-top:.7em;line-height:1.55}
</style>

2.4T 파라미터 MoE 모델을 B300 GPU로 서빙하면서 두 가지 구성을 같은 조건으로 비교했다. 하나는 모델을 NVFP4(4비트)로 압축해 GPU 8장짜리 한 노드에 올린 구성이고, 다른 하나는 FP8(8비트)을 유지한 채 GPU 16장을 두 노드에 나눠 올린 구성이다.

결과가 두 갈래로 갈렸다. 입력을 한 번에 읽어 들이는 **프리필** 단계에서는 GPU를 절반만 쓴 한 노드 구성이 4.3배 빨랐다. 반면 답변을 한 토큰씩 만들어내는 **디코드** 단계에서는 두 구성의 차이가 거의 사라졌다.

하드웨어는 그대로인데 어떤 작업을 시키느냐에 따라 결론이 뒤집힌 셈이다. 왜 그런지 따라가 본다.

## 두 구성의 정밀도가 서로 다른 이유

먼저 짚어둘 것이 있다. 한 노드 쪽은 4비트, 두 노드 쪽은 8비트다. 비교 조건에 변수를 두 개 섞은 것처럼 보이지만, 사실 고를 수 있는 조합이 아니었다.

모델 가중치가 차지하는 용량은 정밀도에 그대로 비례한다.

| 정밀도 | 가중치 총량 | GPU 8장 한 노드에 |
|---|---|---|
| FP8 (1 byte) | 2,400 GB | 들어가지 않음 |
| NVFP4 (0.5 byte) | 1,200 GB | 들어감 |

B300 한 장의 메모리가 288 GB이니 8장이면 2,304 GB다. FP8은 2,400 GB가 필요하다. **96 GB가 부족하다.** KV 캐시와 중간 계산값을 빼고 가중치만 계산해도 그렇다.

그래서 선택지가 두 개뿐이다. 한 노드에 담으려면 4비트로 압축해야 하고, 8비트를 유지하려면 노드를 늘려야 한다.

그런데 두 구성을 나란히 놓으면 조건이 우연히 맞아떨어진다.

| 구성 | 가중치 총량 | GPU 수 | GPU 한 장당 |
|---|---|---|---|
| 한 노드 · NVFP4 | 1,200 GB | 8 | **150 GB** |
| 두 노드 · FP8 | 2,400 GB | 16 | **150 GB** |

정밀도를 절반으로 줄이면서 GPU 수도 절반으로 줄이니, GPU 한 장이 지는 부담이 같아진다. 남는 메모리, 즉 KV 캐시에 쓸 수 있는 공간도 같다는 뜻이다. 실제로 두 구성 모두 동시에 처리할 수 있는 시퀀스 수가 64로 잡혔다.

덕분에 두 구성의 차이가 하나로 좁혀진다. **전문가(Expert)를 GPU에 나눠 배치하는 EP가 노드 경계를 넘느냐 아니냐**다. 한 노드 구성은 EP가 8장 안에서 끝나 GPU끼리 NVLink로 직접 주고받고, 두 노드 구성은 EP가 16장에 걸쳐 있어 상당 부분이 이더넷을 건너간다.

측정 구성은 이렇다.

| | 한 노드 | 두 노드 |
|---|---|---|
| GPU | B300 × 8 | B300 × 16 |
| 모델 | Qwen3.8-2.4T-A95B-NVFP4 | Qwen3.8-2.4T-A95B-FP8 |
| 가중치 정밀도 | NVFP4 (4비트) | FP8 (8비트) |
| EP 범위 | 8 (노드 안) | 16 (노드 간) |
| 병렬 조합 | `TP8/DP1`, `TP4/DP2` | `TP4/DP4`, `TP8/DP2` |
| 노드 간 경로 | 해당 없음 | 리프 스위치 1홉 |

`TP`는 모델 하나를 여러 GPU에 쪼개 얹는 텐서 병렬, `DP`는 같은 모델을 여러 벌 띄워 요청을 나눠 받는 데이터 병렬이다. `TP4/DP4`라면 GPU 4장씩 묶은 복제본이 4벌이라는 뜻이다.

워크로드는 입력 128~16,384 토큰, 출력 1~2,048 토큰, 동시 요청 8~128개 범위에서 조건을 바꿔가며 측정했다. 프리필만 도는 경우, 디코드만 도는 경우, 둘이 섞이는 경우를 각각 따로 돌렸다.

## 프리필: GPU를 절반만 쓴 쪽이 4배 빨랐다

프리필은 사용자가 넣은 입력 전체를 한 번에 읽어 들이는 단계다. 입력 16,384 토큰 조건에서 초당 몇 토큰을 처리하는지 재봤다.

<figure class="fig">
<svg viewBox="0 0 720 236" width="720" height="236" role="img" aria-label="프리필 처리량. 입력 16,384 토큰 조건에서 초당 처리한 토큰 수. 두 계열은 같은 눈금이다." xmlns="http://www.w3.org/2000/svg">
<rect x="46.0" y="4.0" width="10.0" height="10.0" rx="2" fill="#2a78d6" fill-opacity="1.0"/><text x="61.0" y="12.0" fill="currentColor" fill-opacity="0.75" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="400" style="font-variant-numeric:tabular-nums">한 노드 · GPU 8장</text><rect x="168.8" y="4.0" width="10.0" height="10.0" rx="2" fill="#eb6834" fill-opacity="1.0"/><text x="183.8" y="12.0" fill="currentColor" fill-opacity="0.75" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="400" style="font-variant-numeric:tabular-nums">두 노드 · GPU 16장</text>
<text x="38.0" y="22.0" fill="currentColor" fill-opacity="0.45" font-size="9.5" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">동시 요청</text>
<line x1="46.0" y1="26.0" x2="46.0" y2="216.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="46.0" y="230.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">0</text>
<line x1="159.4" y1="26.0" x2="159.4" y2="216.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="159.4" y="230.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">10k</text>
<line x1="272.8" y1="26.0" x2="272.8" y2="216.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="272.8" y="230.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">20k</text>
<line x1="386.2" y1="26.0" x2="386.2" y2="216.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="386.2" y="230.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">30k</text>
<line x1="499.6" y1="26.0" x2="499.6" y2="216.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="499.6" y="230.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">40k</text>
<line x1="613.0" y1="26.0" x2="613.0" y2="216.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="613.0" y="230.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">50k</text>
<text x="38.0" y="48.0" fill="currentColor" fill-opacity="0.8" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">8</text>
<rect x="46.0" y="30.0" width="328.5" height="15.0" rx="2" fill="#2a78d6" fill-opacity="1.0"/>
<text x="381.5" y="41.5" fill="currentColor" fill-opacity="0.9" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">28,970</text>
<rect x="46.0" y="49.0" width="98.3" height="15.0" rx="2" fill="#eb6834" fill-opacity="1.0"/>
<text x="151.3" y="60.5" fill="currentColor" fill-opacity="0.9" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">8,671</text>
<text x="38.0" y="98.0" fill="currentColor" fill-opacity="0.8" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">16</text>
<rect x="46.0" y="80.0" width="590.3" height="15.0" rx="2" fill="#2a78d6" fill-opacity="1.0"/>
<text x="643.3" y="91.5" fill="currentColor" fill-opacity="0.9" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">52,051</text>
<rect x="46.0" y="99.0" width="131.3" height="15.0" rx="2" fill="#eb6834" fill-opacity="1.0"/>
<text x="184.3" y="110.5" fill="currentColor" fill-opacity="0.9" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">11,579</text>
<text x="38.0" y="148.0" fill="currentColor" fill-opacity="0.8" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">32</text>
<rect x="46.0" y="130.0" width="592.0" height="15.0" rx="2" fill="#2a78d6" fill-opacity="1.0"/>
<text x="645.0" y="141.5" fill="currentColor" fill-opacity="0.9" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">52,200</text>
<rect x="46.0" y="149.0" width="138.3" height="15.0" rx="2" fill="#eb6834" fill-opacity="1.0"/>
<text x="191.3" y="160.5" fill="currentColor" fill-opacity="0.9" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">12,198</text>
<text x="38.0" y="198.0" fill="currentColor" fill-opacity="0.8" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">64</text>
<rect x="46.0" y="180.0" width="329.8" height="15.0" rx="2" fill="#2a78d6" fill-opacity="1.0"/>
<text x="382.8" y="191.5" fill="currentColor" fill-opacity="0.9" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">29,080</text>
<rect x="46.0" y="199.0" width="117.5" height="15.0" rx="2" fill="#eb6834" fill-opacity="1.0"/>
<text x="170.5" y="210.5" fill="currentColor" fill-opacity="0.9" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">10,363</text>
</svg>
<figcaption>프리필 처리량. 입력 16,384 토큰 조건에서 초당 처리한 토큰 수. 두 계열은 같은 눈금이다.</figcaption>
</figure>

동시 요청 32개에서 52,200 대 12,198 tok/s, **4.3배** 차이다.

측정한 모든 지점에서 2.8~4.5배 차이가 났다. 양쪽 다 아직 여유가 있는 동시 요청 8~32개 구간으로 좁히면 3.3~4.5배다. 동시 요청 64개에서 차이가 2.8배로 줄어든 것은 두 노드 쪽이 따라잡아서가 아니라, 한 노드 쪽이 먼저 포화에 도달했기 때문이다. 한 노드 구성도 52,200에서 29,080으로 떨어진다.

## 디코드: 차이가 거의 사라진다

디코드는 답변을 한 토큰씩 이어 붙여 만들어내는 단계다. 같은 하드웨어로 이번엔 출력 2,048 토큰을 생성하게 해봤다.

<figure class="fig">
<svg viewBox="0 0 720 186" width="720" height="186" role="img" aria-label="디코드 처리량. 입력 128 / 출력 2,048 토큰 조건에서 초당 생성한 토큰 수. 동시 요청 128개에서 두 구성이 만난다." xmlns="http://www.w3.org/2000/svg">
<rect x="46.0" y="4.0" width="10.0" height="10.0" rx="2" fill="#2a78d6" fill-opacity="1.0"/><text x="61.0" y="12.0" fill="currentColor" fill-opacity="0.75" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="400" style="font-variant-numeric:tabular-nums">한 노드 · GPU 8장</text><rect x="168.8" y="4.0" width="10.0" height="10.0" rx="2" fill="#eb6834" fill-opacity="1.0"/><text x="183.8" y="12.0" fill="currentColor" fill-opacity="0.75" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="400" style="font-variant-numeric:tabular-nums">두 노드 · GPU 16장</text>
<text x="38.0" y="22.0" fill="currentColor" fill-opacity="0.45" font-size="9.5" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">동시 요청</text>
<line x1="46.0" y1="26.0" x2="46.0" y2="166.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="46.0" y="180.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">0</text>
<line x1="178.2" y1="26.0" x2="178.2" y2="166.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="178.2" y="180.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">500</text>
<line x1="310.4" y1="26.0" x2="310.4" y2="166.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="310.4" y="180.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">1k</text>
<line x1="442.6" y1="26.0" x2="442.6" y2="166.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="442.6" y="180.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">1.5k</text>
<line x1="574.8" y1="26.0" x2="574.8" y2="166.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="574.8" y="180.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">2k</text>
<text x="38.0" y="48.0" fill="currentColor" fill-opacity="0.8" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">32</text>
<rect x="46.0" y="30.0" width="388.1" height="15.0" rx="2" fill="#2a78d6" fill-opacity="1.0"/>
<text x="441.1" y="41.5" fill="currentColor" fill-opacity="0.9" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">1,468</text>
<rect x="46.0" y="49.0" width="172.9" height="15.0" rx="2" fill="#eb6834" fill-opacity="1.0"/>
<text x="225.9" y="60.5" fill="currentColor" fill-opacity="0.9" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">654</text>
<text x="38.0" y="98.0" fill="currentColor" fill-opacity="0.8" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">64</text>
<rect x="46.0" y="80.0" width="592.0" height="15.0" rx="2" fill="#2a78d6" fill-opacity="1.0"/>
<text x="645.0" y="91.5" fill="currentColor" fill-opacity="0.9" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">2,239</text>
<rect x="46.0" y="99.0" width="336.1" height="15.0" rx="2" fill="#eb6834" fill-opacity="1.0"/>
<text x="389.1" y="110.5" fill="currentColor" fill-opacity="0.9" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">1,271</text>
<text x="38.0" y="148.0" fill="currentColor" fill-opacity="0.8" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">128</text>
<rect x="46.0" y="130.0" width="590.4" height="15.0" rx="2" fill="#2a78d6" fill-opacity="1.0"/>
<text x="643.4" y="141.5" fill="currentColor" fill-opacity="0.9" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">2,233</text>
<rect x="46.0" y="149.0" width="576.4" height="15.0" rx="2" fill="#eb6834" fill-opacity="1.0"/>
<text x="629.4" y="160.5" fill="currentColor" fill-opacity="0.9" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">2,180</text>
</svg>
<figcaption>디코드 처리량. 입력 128 / 출력 2,048 토큰 조건에서 초당 생성한 토큰 수. 동시 요청 128개에서 두 구성이 만난다.</figcaption>
</figure>

동시 요청 128개에서 2,233 대 2,180. **1.02배**로 사실상 같다.

하드웨어는 그대로인데 작업 성격만 바뀌었더니 결과가 4.3배에서 1.02배로 변했다. 그렇다면 차이를 만든 것은 하드웨어 자체가 아니라, 그 작업이 하드웨어의 어느 부분을 쓰는지다.

## 4.3배 중 얼마가 노드를 나눈 대가인가

4.3배를 전부 노드를 나눈 대가로 볼 수는 없다. NVFP4는 FP8보다 가중치를 절반만 읽으면 되니, 노드 수와 관계없이 그 자체로 빠르다. 이 몫을 먼저 걷어내야 한다.

디코드 구간이 기준이 된다. 디코드는 메모리에서 가중치를 읽어오는 속도가 성능을 좌우하고 노드 간 통신량은 적다. 그러니 여기서 벌어지는 차이는 대부분 정밀도에서 온다고 볼 수 있다.

토큰 하나를 만드는 데 걸린 시간을 보면 이렇다.

| 동시 요청 | 한 노드 | 두 노드 | 배율 |
|---|---|---|---|
| 32개 | 21.60 ms | 47.64 ms | 2.21배 |
| 64개 | 28.36 ms | 49.97 ms | 1.76배 |
| 128개 | 28.44 ms | 58.02 ms | 2.04배 |

일관되게 약 2배다. 읽어야 할 가중치가 절반이 된 효과와 맞아떨어진다.

정밀도가 설명하는 몫을 2배로 놓으면, 프리필의 3.3~4.5배에서 남는 **약 2배가 노드를 나눈 대가**라는 계산이 나온다.

다만 이건 추정이다. 디코드의 2배는 메모리에서 읽는 양이 줄어든 결과이고, 프리필에서 4비트가 얻는 이득은 연산량이 줄어든 결과다. 원리가 다른데 배율이 비슷하게 나온 것은 NVFP4가 두 축을 함께 개선하는 포맷이기 때문이지, 반드시 그래야 할 이유는 없다. 정확히 가르려면 한 노드에 FP8을 올린 대조군이 필요한데, 앞에서 봤듯 그 구성은 메모리에 들어가지 않는다.

## 노드 사이에는 얼마나 흘렀나

두 노드 구성에서 실제로 오간 트래픽을 스위치 인터페이스 카운터로 확인했다.

<figure class="fig">
<svg viewBox="0 0 720 188" width="720" height="188" role="img" aria-label="노드 간 링크 사용률. 회색 막대 전체가 400 Gbps 용량이고, 색칠된 부분이 실제로 오간 양이다." xmlns="http://www.w3.org/2000/svg">
<text x="132.0" y="12.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="400" style="font-variant-numeric:tabular-nums">0</text>
<text x="602.0" y="12.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">400 Gbps (링크 용량)</text>
<text x="122.0" y="34.0" fill="currentColor" fill-opacity="0.85" font-size="11" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">두 노드 TP4/DP4</text>
<text x="122.0" y="48.0" fill="currentColor" fill-opacity="0.5" font-size="10" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">프리필</text>
<rect x="132.0" y="22.0" width="470.0" height="18.0" rx="2" fill="currentColor" fill-opacity="0.07"/>
<rect x="132.0" y="22.0" width="31.2" height="18.0" rx="2" fill="#eb6834" fill-opacity="1"/>
<text x="612.0" y="30.0" fill="currentColor" fill-opacity="0.9" font-size="11.5" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">6.64%</text>
<text x="612.0" y="43.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="400" style="font-variant-numeric:tabular-nums">26.6 Gbps</text>
<text x="122.0" y="74.0" fill="currentColor" fill-opacity="0.85" font-size="11" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">두 노드 TP8/DP2</text>
<text x="122.0" y="88.0" fill="currentColor" fill-opacity="0.5" font-size="10" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">프리필</text>
<rect x="132.0" y="62.0" width="470.0" height="18.0" rx="2" fill="currentColor" fill-opacity="0.07"/>
<rect x="132.0" y="62.0" width="32.9" height="18.0" rx="2" fill="#eb6834" fill-opacity="1"/>
<text x="612.0" y="70.0" fill="currentColor" fill-opacity="0.9" font-size="11.5" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">7.00%</text>
<text x="612.0" y="83.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="400" style="font-variant-numeric:tabular-nums">28.0 Gbps</text>
<text x="122.0" y="114.0" fill="currentColor" fill-opacity="0.85" font-size="11" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">두 노드 TP4/DP4</text>
<text x="122.0" y="128.0" fill="currentColor" fill-opacity="0.5" font-size="10" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">디코드</text>
<rect x="132.0" y="102.0" width="470.0" height="18.0" rx="2" fill="currentColor" fill-opacity="0.07"/>
<rect x="132.0" y="102.0" width="10.5" height="18.0" rx="2" fill="#eb6834" fill-opacity="0.5"/>
<text x="612.0" y="110.0" fill="currentColor" fill-opacity="0.9" font-size="11.5" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">2.23%</text>
<text x="612.0" y="123.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="400" style="font-variant-numeric:tabular-nums">8.9 Gbps</text>
<text x="122.0" y="154.0" fill="currentColor" fill-opacity="0.85" font-size="11" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">두 노드 TP8/DP2</text>
<text x="122.0" y="168.0" fill="currentColor" fill-opacity="0.5" font-size="10" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">디코드</text>
<rect x="132.0" y="142.0" width="470.0" height="18.0" rx="2" fill="currentColor" fill-opacity="0.07"/>
<rect x="132.0" y="142.0" width="13.2" height="18.0" rx="2" fill="#eb6834" fill-opacity="0.5"/>
<text x="612.0" y="150.0" fill="currentColor" fill-opacity="0.9" font-size="11.5" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">2.80%</text>
<text x="612.0" y="163.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="400" style="font-variant-numeric:tabular-nums">11.2 Gbps</text>
</svg>
<figcaption>노드 간 링크 사용률. 회색 막대 전체가 400 Gbps 용량이고, 색칠된 부분이 실제로 오간 양이다.</figcaption>
</figure>

프리필 구간이 디코드 구간보다 2.5~3배 더 흘린다. 서버 한 대의 업링크 16개를 합치면 프리필 425~448 Gbps, 디코드 143~179 Gbps다.

이유는 단순하다. 프리필은 한 번에 수천 개의 토큰을 전문가들에게 나눠 보내고, 디코드는 요청당 한 토큰만 보낸다. 오가는 양이 그 순간 처리 중인 토큰 수를 그대로 따라간다.

눈에 띄는 건 사용률 자체다. 프리필 처리량이 4배로 벌어지는 바로 그 구간에서, 400G 링크는 7% 안팎만 쓰고 있었다.

### 트래픽이 같아도 결과는 다르다

두 노드 구성 안에서 병렬 조합만 바꿔보면 이 점이 분명해진다.

| 병렬 조합 | 링크 사용률 | 프리필 처리량 |
|---|---|---|
| `TP4/DP4` | 6.64% | 12,198 tok/s |
| `TP8/DP2` | 7.00% | 8,026 tok/s |

노드 사이를 오간 양은 사실상 같은데 처리량은 1.5배 차이가 난다. 차이를 만든 것은 트래픽 양이 아니라, 모델 복제본을 몇 벌로 나눴는지에 따른 배치 구성이다.

## 무엇이 프리필을 붙잡고 있나

MoE 모델은 레이어마다 토큰을 여러 전문가에게 나눠 보내고, 결과를 다시 모은다. 이 왕복이 끝나야 다음 레이어로 넘어간다. 기다리는 동안 다른 일을 할 수 없는 구조다.

EP가 GPU 16장에 걸쳐 있으면 이 왕복의 상당 부분이 노드 경계를 넘고, 그것이 레이어 수만큼 반복된다.

프리필은 한 번에 수천 토큰이 이 경로를 통과하니 왕복 대기가 그대로 쌓인다. 디코드는 요청당 한 토큰이라, 왕복 대기가 어차피 발생하는 메모리 읽기 대기에 묻힌다. 측정된 비대칭이 이 구조와 들어맞는다.

덧붙이면 이번 두 노드 구성은 rail-optimized 설계라, 노드 간 트래픽이 스파인까지 올라가지 않고 리프 스위치 한 홉으로 끝났다. 두 노드로 나눌 때 경로상 가장 유리한 조건이다. 랙을 넘어가는 실제 환경이라면 이보다 나아지기는 어렵다.

정리하면, 한 노드 구성이 앞선 것은 4비트로 압축해서만이 아니다. **압축한 덕분에 EP 전체가 NVLink 안에 들어왔기 때문**이다. 이 비교가 실제로 잰 것은 양자화와 분산의 우열이 아니라, 전문가 분산을 한 노드 안에 가둬둘 때의 이득에 가깝다.

## 한 노드 구성에도 한계는 있다

한 노드가 언제나 낫다는 뜻은 아니다. 가장 빠른 조합인 `TP8/DP1`에 동시 요청을 8개에서 128개까지 늘려봤다.

<figure class="fig">
<svg viewBox="0 0 720 188" width="720" height="188" role="img" aria-label="한 노드 TP8/DP1 구성. 처리량 곡선은 완만하게 꺾이지만 첫 토큰 지연은 자릿수가 바뀐다." xmlns="http://www.w3.org/2000/svg">
<text x="0.0" y="13.0" fill="currentColor" fill-opacity="0.9" font-size="11.5" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">초당 생성 토큰</text>
<line x1="46.0" y1="138.0" x2="333.0" y2="138.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="38.0" y="141.5" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">0</text>
<line x1="46.0" y1="111.0" x2="333.0" y2="111.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="38.0" y="114.5" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">500</text>
<line x1="46.0" y1="84.0" x2="333.0" y2="84.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="38.0" y="87.5" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">1k</text>
<line x1="46.0" y1="57.0" x2="333.0" y2="57.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="38.0" y="60.5" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">1.5k</text>
<line x1="46.0" y1="30.0" x2="333.0" y2="30.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="38.0" y="33.5" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">2k</text>
<polyline points="46.0,111.3 117.8,92.1 189.5,70.7 261.2,41.9 333.0,55.5" fill="none" stroke="#2a78d6" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
<circle cx="46.0" cy="111.3" r="3.6" fill="#2a78d6"/>
<text x="46.0" y="156.0" fill="currentColor" fill-opacity="0.6" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">8</text>
<circle cx="117.8" cy="92.1" r="3.6" fill="#2a78d6"/>
<text x="117.8" y="156.0" fill="currentColor" fill-opacity="0.6" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">16</text>
<circle cx="189.5" cy="70.7" r="3.6" fill="#2a78d6"/>
<text x="189.5" y="156.0" fill="currentColor" fill-opacity="0.6" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">32</text>
<circle cx="261.2" cy="41.9" r="3.6" fill="#2a78d6"/>
<text x="261.2" y="156.0" fill="currentColor" fill-opacity="0.6" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">64</text>
<circle cx="333.0" cy="55.5" r="3.6" fill="#2a78d6"/>
<text x="333.0" y="156.0" fill="currentColor" fill-opacity="0.6" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">128</text>
<text x="189.5" y="173.0" fill="currentColor" fill-opacity="0.5" font-size="10" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">동시 요청 수</text>
<text x="373.0" y="13.0" fill="currentColor" fill-opacity="0.9" font-size="11.5" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">첫 토큰 지연 (ms)</text>
<line x1="419.0" y1="138.0" x2="706.0" y2="138.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="411.0" y="141.5" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">0s</text>
<line x1="419.0" y1="113.5" x2="706.0" y2="113.5" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="411.0" y="117.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">5s</text>
<line x1="419.0" y1="88.9" x2="706.0" y2="88.9" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="411.0" y="92.4" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">10s</text>
<line x1="419.0" y1="64.4" x2="706.0" y2="64.4" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="411.0" y="67.9" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">15s</text>
<line x1="419.0" y1="39.8" x2="706.0" y2="39.8" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="411.0" y="43.3" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">20s</text>
<polyline points="419.0,132.4 490.8,132.1 562.5,130.5 634.2,127.9 706.0,37.7" fill="none" stroke="#eb6834" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
<circle cx="419.0" cy="132.4" r="3.6" fill="#eb6834"/>
<text x="419.0" y="156.0" fill="currentColor" fill-opacity="0.6" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">8</text>
<circle cx="490.8" cy="132.1" r="3.6" fill="#eb6834"/>
<text x="490.8" y="156.0" fill="currentColor" fill-opacity="0.6" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">16</text>
<circle cx="562.5" cy="130.5" r="3.6" fill="#eb6834"/>
<text x="562.5" y="156.0" fill="currentColor" fill-opacity="0.6" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">32</text>
<circle cx="634.2" cy="127.9" r="3.6" fill="#eb6834"/>
<text x="634.2" y="156.0" fill="currentColor" fill-opacity="0.6" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">64</text>
<circle cx="706.0" cy="37.7" r="3.6" fill="#eb6834"/>
<text x="706.0" y="156.0" fill="currentColor" fill-opacity="0.6" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">128</text>
<text x="562.5" y="173.0" fill="currentColor" fill-opacity="0.5" font-size="10" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">동시 요청 수</text>
</svg>
<figcaption>한 노드 TP8/DP1 구성. 처리량 곡선은 완만하게 꺾이지만 첫 토큰 지연은 자릿수가 바뀐다.</figcaption>
</figure>

동시 요청 64개에서 초당 1,780 토큰으로 정점을 찍고, 128개에서는 처리량이 오히려 14% 떨어진다. 그 사이 첫 토큰이 나오기까지 걸리는 시간은 2.05초에서 20.4초로 뛴다.

동시에 처리할 수 있는 시퀀스가 64개인데 요청을 128개 밀어 넣었으니, 나머지 절반은 순서를 기다린 셈이다.

처리량만 보고 튜닝하면 이 지점을 놓치기 쉽다. 처리량 곡선은 완만하게 꺾일 뿐이지만, 첫 토큰 지연은 자릿수가 바뀐다.

## 꼬리 지연은 구성에 따라 크게 갈린다

프리필과 디코드 요청이 섞여 들어오면, 프리필 묶음이 GPU를 붙잡고 있는 동안 디코드 토큰이 밀린다. 앞선 작업이 뒤를 막는 head-of-line blocking이다.

이 현상이 구성마다 얼마나 다른지 봤다. 토큰과 토큰 사이의 간격을 잰 값이다.

<figure class="fig">
<svg viewBox="0 0 720 258" width="720" height="258" role="img" aria-label="프리필과 디코드를 섞은 조건, 동시 요청 128개. 중앙값은 26~58 ms로 모여 있지만 p99는 자릿수가 다르다." xmlns="http://www.w3.org/2000/svg">
<rect x="118.0" y="4.0" width="10.0" height="10.0" rx="2" fill="#2a78d6" fill-opacity="1.0"/><text x="133.0" y="12.0" fill="currentColor" fill-opacity="0.75" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="400" style="font-variant-numeric:tabular-nums">중앙값</text><rect x="174.8" y="4.0" width="10.0" height="10.0" rx="2" fill="#1baf7a" fill-opacity="1.0"/><text x="189.8" y="12.0" fill="currentColor" fill-opacity="0.75" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="400" style="font-variant-numeric:tabular-nums">평균</text><rect x="225.0" y="4.0" width="10.0" height="10.0" rx="2" fill="#eb6834" fill-opacity="1.0"/><text x="240.0" y="12.0" fill="currentColor" fill-opacity="0.75" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="400" style="font-variant-numeric:tabular-nums">p99</text>
<line x1="118.0" y1="24.0" x2="118.0" y2="230.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="118.0" y="245.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">0</text>
<line x1="223.4" y1="24.0" x2="223.4" y2="230.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="223.4" y="245.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">500</text>
<line x1="328.8" y1="24.0" x2="328.8" y2="230.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="328.8" y="245.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">1,000</text>
<line x1="434.2" y1="24.0" x2="434.2" y2="230.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="434.2" y="245.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">1,500</text>
<line x1="539.7" y1="24.0" x2="539.7" y2="230.0" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>
<text x="539.7" y="245.0" fill="currentColor" fill-opacity="0.55" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="middle" font-weight="400" style="font-variant-numeric:tabular-nums">2,000 ms</text>
<text x="108.0" y="45.0" fill="currentColor" fill-opacity="0.85" font-size="11" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">한 노드 NVFP4</text>
<text x="108.0" y="59.0" fill="currentColor" fill-opacity="0.5" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">TP8/DP1</text>
<rect x="118.0" y="30.0" width="5.6" height="10.0" rx="2" fill="#2a78d6" fill-opacity="1.0"/>
<rect x="118.0" y="42.0" width="8.9" height="10.0" rx="2" fill="#1baf7a" fill-opacity="1.0"/>
<rect x="118.0" y="54.0" width="49.4" height="10.0" rx="2" fill="#eb6834" fill-opacity="1.0"/>
<text x="176.4" y="50.0" fill="currentColor" fill-opacity="0.9" font-size="11.5" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">234 ms</text>
<text x="176.4" y="62.0" fill="currentColor" fill-opacity="0.55" font-size="10" text-anchor="start" font-weight="400" style="font-variant-numeric:tabular-nums">중앙값 ×9</text>
<text x="108.0" y="97.0" fill="currentColor" fill-opacity="0.85" font-size="11" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">한 노드 NVFP4</text>
<text x="108.0" y="111.0" fill="currentColor" fill-opacity="0.5" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">TP4/DP2</text>
<rect x="118.0" y="82.0" width="9.6" height="10.0" rx="2" fill="#2a78d6" fill-opacity="1.0"/>
<rect x="118.0" y="94.0" width="22.0" height="10.0" rx="2" fill="#1baf7a" fill-opacity="1.0"/>
<rect x="118.0" y="106.0" width="116.2" height="10.0" rx="2" fill="#eb6834" fill-opacity="1.0"/>
<text x="243.2" y="102.0" fill="currentColor" fill-opacity="0.9" font-size="11.5" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">551 ms</text>
<text x="243.2" y="114.0" fill="currentColor" fill-opacity="0.55" font-size="10" text-anchor="start" font-weight="400" style="font-variant-numeric:tabular-nums">중앙값 ×12</text>
<text x="108.0" y="149.0" fill="currentColor" fill-opacity="0.85" font-size="11" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">두 노드 FP8</text>
<text x="108.0" y="163.0" fill="currentColor" fill-opacity="0.5" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">TP4/DP4</text>
<rect x="118.0" y="134.0" width="11.9" height="10.0" rx="2" fill="#2a78d6" fill-opacity="1.0"/>
<rect x="118.0" y="146.0" width="33.5" height="10.0" rx="2" fill="#1baf7a" fill-opacity="1.0"/>
<rect x="118.0" y="158.0" width="369.4" height="10.0" rx="2" fill="#eb6834" fill-opacity="1.0"/>
<text x="496.4" y="154.0" fill="currentColor" fill-opacity="0.9" font-size="11.5" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">1,752 ms</text>
<text x="496.4" y="166.0" fill="currentColor" fill-opacity="0.55" font-size="10" text-anchor="start" font-weight="400" style="font-variant-numeric:tabular-nums">중앙값 ×31</text>
<text x="108.0" y="201.0" fill="currentColor" fill-opacity="0.85" font-size="11" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">두 노드 FP8</text>
<text x="108.0" y="215.0" fill="currentColor" fill-opacity="0.5" font-size="10" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="end" font-weight="400" style="font-variant-numeric:tabular-nums">TP8/DP2</text>
<rect x="118.0" y="186.0" width="12.1" height="10.0" rx="2" fill="#2a78d6" fill-opacity="1.0"/>
<rect x="118.0" y="198.0" width="48.0" height="10.0" rx="2" fill="#1baf7a" fill-opacity="1.0"/>
<rect x="118.0" y="210.0" width="469.5" height="10.0" rx="2" fill="#eb6834" fill-opacity="1.0"/>
<text x="596.5" y="206.0" fill="currentColor" fill-opacity="0.9" font-size="11.5" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" text-anchor="start" font-weight="600" style="font-variant-numeric:tabular-nums">2,227 ms</text>
<text x="596.5" y="218.0" fill="currentColor" fill-opacity="0.55" font-size="10" text-anchor="start" font-weight="400" style="font-variant-numeric:tabular-nums">중앙값 ×39</text>
</svg>
<figcaption>프리필과 디코드를 섞은 조건, 동시 요청 128개. 중앙값은 26~58 ms로 모여 있지만 p99는 자릿수가 다르다.</figcaption>
</figure>

| 구성 | 중앙값 | 평균 | p99 | p99 / 중앙값 |
|---|---|---|---|---|
| 한 노드 `TP8/DP1` | 26.7 ms | 42.1 ms | 234.3 ms | **8.8배** |
| 한 노드 `TP4/DP2` | 45.5 ms | 104.3 ms | 551.3 ms | 12.1배 |
| 두 노드 `TP4/DP4` | 56.5 ms | 159.1 ms | 1,752.0 ms | 31.0배 |
| 두 노드 `TP8/DP2` | 57.6 ms | 227.9 ms | 2,226.7 ms | 38.7배 |

여기서 p99는 가장 느렸던 1%가 겪은 값이다. 100번 중 한 번은 이만큼 기다렸다는 뜻이다.

head-of-line blocking 자체는 어느 구성에서도 없어지지 않는다. 한 노드 구성에서도 중앙값의 8.8배가 남는다. 다만 234 ms와 1,752 ms는 성격이 다른 숫자다. 앞쪽은 튜닝으로 다뤄볼 여지가 있고, 뒤쪽은 대화형 서비스로 쓰기 어렵다.

그리고 네 구성 모두 중앙값만 보면 26~58 ms로 비슷하다. 평균과 중앙값만 봐서는 이 차이가 드러나지 않는다.

## KV 캐시 재사용은 효과가 없었다

모든 조건을 KV 캐시 재사용을 켠 경우와 끈 경우로 나눠 돌렸는데, 의미 있는 차이가 나온 구간이 없다.

| 구성 (동시 요청 128개) | 껐을 때 | 켰을 때 |
|---|---|---|
| 두 노드 `TP4/DP4` | 752.93 tok/s | 750.64 tok/s |
| 한 노드 `TP8/DP1` | 1,527.92 tok/s | 1,559.45 tok/s |

측정 오차 범위다. 이유는 명확하다. 테스트에 쓴 데이터는 요청마다 앞부분이 제각각이라, 재사용할 공통 부분이 애초에 없었다.

기능을 켠 것과 기능이 실제로 동작한 것은 다르다. 이 효과를 보려면 같은 시스템 프롬프트가 반복되는 데이터를 따로 만들어야 한다. 그렇게 하지 않은 측정에 "캐시 재사용을 켜고 쟀다"고 적는 것은 아무 정보도 담지 못한다.

## 확인하지 못한 것

- **정밀도와 노드 확장을 정확히 가르지는 못했다.** 한 노드에 FP8을 올린 대조군이 필요한데 그 구성은 메모리에 들어가지 않는다. "약 2배 대 약 2배"는 디코드를 기준으로 삼은 추정이다.
- **링크 사용률은 10초 평균이다.** 이 간격으로는 순간적으로 트래픽이 몰렸는지를 볼 수 없다. 평균이 낮았다는 것이 링크에 여유가 있었다는 뜻인지, 짧게 몰렸다 빠지기를 반복했다는 뜻인지는 이 데이터로 구분되지 않는다.
- **4비트 압축이 답변 품질에 주는 영향은 재지 않았다.** 이 글은 속도만 다룬다.
- **노드를 더 늘렸을 때는 확인하지 못했다.** 이번 측정은 두 노드까지다. 프리필 손실이 노드 수에 비례해 커지는지, 어느 지점에서 완만해지는지는 따로 재봐야 한다.

## 요약

1. 2.4T 파라미터 모델은 FP8로는 GPU 8장 한 노드에 들어가지 않는다. 2,304 GB 대 2,400 GB로 96 GB가 부족하다.
2. 그래서 선택지가 한 노드 NVFP4와 두 노드 FP8 두 가지뿐이고, 두 구성의 GPU당 가중치 부담은 150 GB로 같다.
3. 프리필은 한 노드 구성이 3.3~4.5배 빠르다. 입력 16,384 토큰, 동시 요청 32개에서 52,200 대 12,198 tok/s.
4. 동시 요청 64개에서 차이가 2.8배로 좁혀지는데, 두 노드가 나아진 것이 아니라 한 노드가 먼저 포화에 도달했기 때문이다.
5. 디코드는 동시 요청 128개에서 2,233 대 2,180으로 차이가 거의 없다.
6. 디코드에서 일관되게 나온 약 2배를 정밀도 몫으로 놓으면, 프리필에 남는 약 2배가 노드를 나눈 대가로 추정된다.
7. 노드 간 링크 사용률은 프리필 6.6~7.0%, 디코드 2.2~2.8%였다. 오가는 양이 그 순간 처리 중인 토큰 수를 따라간다.
8. 두 노드 안에서 병렬 조합만 바꾸면 링크 사용률은 같은데 프리필 처리량이 1.5배 차이 난다. 트래픽 양이 아니라 배치 구성의 영향이다.
9. 한 노드 구성에도 한계는 있다. 동시 요청 64개를 넘기면 처리량은 14% 떨어지고 첫 토큰 지연은 2.05초에서 20.4초로 늘어난다.
10. head-of-line blocking은 한 노드에서도 남는다. p99가 중앙값의 8.8배이고, 두 노드에서는 31배까지 벌어진다.

## 부록: 측정값

| 구성 | 프리필 tok/s<br>입력 16,384 · 동시 32 | 디코드 tok/s<br>출력 2,048 · 동시 128 | 기본 조건 tok/s<br>동시 64 | 첫 토큰 지연 ms<br>동시 64 |
|---|---|---|---|---|
| 한 노드 NVFP4 `TP8/DP1` | 52,200 | 2,233 | 1,780 | 2,053 |
| 한 노드 NVFP4 `TP4/DP2` | 19,624 | 2,526 | 932 | 3,931 |
| 두 노드 FP8 `TP4/DP4` | 12,198 | 2,180 | 625 | 10,301 |
| 두 노드 FP8 `TP8/DP2` | 8,026 | 2,077 | 508 | 16,774 |

모든 행은 동시 처리 시퀀스 상한 64, KV 캐시 재사용을 끈 조건이다. 기본 조건은 입력 4,096 / 출력 512 토큰이다.
