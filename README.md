# Netadapt

##Jetson Xavier AGX에서 Netadapt 구현 및 비교

Netadapt를 적용한 엣지디바이스와 GPU 
(Jetson Xavier AGX vs RTX-A6000)에서의 실제 Latency 비교 분석.

1. 모델은 포항공대/서울대에서 협업으로 만든 efficientnet 2종류와 mobileNet_V2를 사용
2. Xavier AGX에서 LookUpTable(lut)을 생성
3. 서버 GPU에서 Channel pruning 진행
4. resume하여 pruning ratio를 정하고 LUT에서의 latency와 실제 latency를 비교 및 정확도 확인
