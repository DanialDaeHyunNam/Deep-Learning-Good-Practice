## Tensorflow practice
### Overview
- DNN(2019. 1.30)
  가장 기본적인 Deep Neural Network를 설계해봤습니다. 목적은 Graph를 만드는 Model 클래스와 Train 함수를 구분하여, 여러 개의 모델에 대해서 똑같은 Train을 할 수 있는 구조를 만들어 봤습니다.
  <br>
  이는 "각 모듈은 한 가지의 역할 또는 책임을 가져야 한다(단일 책임의 원칙)"을 따르는 의도로써, Model의 경우는 Session을 외부로부터 받거나 내부적으로 선언하지 않고 오직 Graph를 선언하는 역할을 하고, Train에서 Session을 통해서 학습하는 방법입니다.
