## **processminor.py** summarize and Keypoint

**레이블링**

- 레이블이 현재 고장 시점에만 1로 되어있음
- 여기서 타임라인은 2분간격임
- 여기서는 이를 다시 재 레이블링함.
  - 10분에 고장나서 1이면, 10분 row를 없애고, 고장 임박 4분전 부터 레이블을 1로바꿈
  - 예)6분, 8분 row에서 레이블 = 1



**2D to 3D array**

- lstm 형태로 만들기 위해서 2d를 3d array로 변환

- shape = (sample, lookback, features)

  - sample : 관측치 수, 데이터 수, number of Rows
  - lookback : LSTM model에서 과거 어디까지 볼 것인가에 대한것.
  - feartures : 현재 인풋으로 사용할 갯수

- ```python
  def temporalize(): 함수를 통해, Lookback 차원을 추가
  ```

- 우리 phm 코드로 치면, sequence length가 여기서 lookback임



**Standardize the Data**

> 대원칙 :
>
> 	train 데이터와 test 데이터를 분리하기 전에 normalize하면 안됨
> 	
> 	test 데이터는 없는 데이터라 생각하는 게 맞으니깐.

```
전체 데이터 - > 분류 - > train 표준화 - > 학습
test는 train의 평균과 잔차값을 이용하여 standardize.실시
```

- 저자는 2d array에서 하지 않고, 3d array로 만든 뒤 normalize 실시함. 
  - 2d에서 해버리면 validation test를 실행할 수 없음.