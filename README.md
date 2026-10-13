# Implement transformer from scratch
## utils.py
- declare hyperparameters in model
- scaled_dot_product

## layers.py
- Multi_Head_Attention: all of 3 types (self, masked, cross)
- RMSNorm: Layer Normalization
- SwiGLUFFN: Feed-Forward layer using activation function SwiGLU 

## encoder_decoder.py
- EncoderLayer: 1 block in Encoder
- DecoderLayer: 1 block in Decoder
- Encoder: Encoder by N blocks
- Decoder: Decoder by N blocks

## transformer.py
- Transformer

## preprocess.py
1. 데이터셋 로드 (datasets.load_dataset)
2. 샘플 구조 확인 (translation["de"], translation["en"])
3. 토크나이저 선택 및 로드 (AutoTokenizer.from_pretrained(...))
4. 전처리 함수 작성 (de 입력 / en 타깃 tokenization)
5. map(batched=True)로 전체 토큰화
6. 패딩 전략 결정 (collate_fn에서 batch마다 패딩)
7. DataLoader 구성
8. (커스텀 Transformer용) decoder 입력/정답 shift + mask 생성
9. 배치 shape sanity check
10. 그 다음에 모델 forward/train loop