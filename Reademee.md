## MobileBERT를 활용한 와인 리뷰 분석 프로젝트
badge icon 참고 사이트<br>
<img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?&style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white" />


<!--# 개요 A4용지로 1장 정도 (그림포함)-->

## 1. 개요
![wn](https://github.com/qkrtnals/-/assets/79901070/8c1e29f2-45b0-482b-bfa4-2c82d0d9c7bd)

### 1.1 문제정의
IWSR(International Wine & Spirits Trend)은 런던 소재 시장 분석 회사로 와인 및 스피릿 시장 연구를 위해 300명의 애널리스트들이 
전 세계 수입상들을 만나 인터뷰를 통해 얻은 조사 결과다. 
전 세계 와인시장은 2013년경 중국 정부의 반부패 정책으로 중국 와인 시장이 위축되면서 타격을 받은 적이 있으나 지금은 회복세를 보이고 있다.
향후 5년의 주류 시장 미래 동향을 진단했다는 것이 특징이다.

이 프로젝트에서는 와인을 먹어본 소비자들의 와인 리뷰, 평점 등 다양한 특징에 따라 긍정 또는 부정을 예측하는 인공지능 모델을 개발하고자 한다.






### 1.2 와인의 영향력
<!--대한민국에 와인 열풍이 거세게 불면서 와인이 주류(酒類) 시장의 주류(主流)로 자리잡고 있다.
과거 상류층의 기호식품으로 여겨지던 와인은 코로나19 바이러스 사태를 계기로 홈술족이 크게 늘고, 
수입 주류업계를 옥죄던 일부 규제가 풀리면서 소주, 맥주만큼이나 일상 생활 속으로 파고들며 대중화되고 있다. 
때아닌 와인 수요 훈풍에 와인 수입액은 최고치를 기록했고 와인 수입사들은 기지개를 켜며 상장 작업에 착수하고 있다. 
국내 와인 수입 1위 업체(신세계L&B)를 보유한 신세계그룹은 최근 미국 나파밸리 프리미엄 와이너리의 주인이 됐다.
-->

http://www.foodicon.co.kr 기사에 따르면
한국의 와인시장은 일본에 비해 소비가 크게 증가하고 있으며 향후에도 계속될 것으로 예상된다. 
2011~2021년 한국의 와인 소비량은 약 45%(스틸와인 363만 케이스) 정도 증가할 것이다. 
여기에 스파클링와인을 포함한 전체 와인시장은 2021년경 407만 케이스 정도에 달할 것으로 예상된다. 
한국 시장의 흥미로운 점은 스틸와인시장과 스파클링와인시장 모두 성장하고 있다.

## 2. 데이터
### 2.1 원시 데이터
데이터에 대한 소개
https://www.kaggle.com/datasets/samuelmcguire/wine-reviews-data

총 데이터 건수는 총 323237건이며
칼럼의 종류에는 
와인 이름, 와이너리 이름, 카테고리: 와인 종류(예: 레드, 화이트, 스파클링), 지정, 품종 : 포도의 종류, 
아펠라시옹(Appellation): 와인이 생산되는 지역, 알코올 함량, 가격, 평가, 리뷰어 이름, 검토가 있다.

|wine | winery | category | designation | varietal | appellation | alcohol | price | rating | reviewer | review |
|--|--|--|--|--|--|--|--|--|--|--|
|와인명 | 생산지 | 종류 | 지역 | 품종 | 지역명 | 도수 | 가격 | 평점 | 리뷰어 | 리뷰 |



### 2.2 추출한 데이터
화이트 와인 데이터는 95424건이며 평점(rating)은 80점부터 100점까지 구성되어있다.<br/>

|-| wine |	winery |	category |	designation |	varietal |	appellation |	alcohol |	price |	rating |	reviewer |	review |
|-|--|--|--|--|--|--|--|--|--|--|--|
|4|	Tenuta San Francesco 2007 Tramonti White (Camp...|	Tenuta San Francesco|	White|	Tramonti|	White Blend|	Campania, Southern Italy, Italy|	13.5%|	$21|	85|	NaN|	This intriguing blend of Falanghina, Biancolel...|
|10|	Merry Edwards 2011 Sauvignon Blanc (Russian Ri...|	Merry Edwards|	White|	NaN|	Sauvignon Blanc|	Russian River Valley, Sonoma, California, US|	14.1%|	$32|	88|	NaN|	Despite a chilly vintage, the winery successfu...|
|15	|Jidvei 2015 Demisec Gewurztraminer (Jidvei)|	Jidvei|	White|	Demisec|	Gewürztraminer, Gewürztraminer|	Jidvei, Romania	|12.5%|	$10|	86|	Jeff Jenssen|	This semisweet wine has a flowery bouquet of j...|
|20	|Winery| of Good Hope 2011 Bush Vine Chenin Blan...|	Winery of Good Hope|	White|	Bush Vine|	Chenin Blanc|	Stellenbosch, South Africa|	13%|	$12|	86|	Lauren Buzzeo	|This wine shows good balance between the livel...|
|26	|Sallier de la Tour 2017 Inzolia (Sicilia)|	Sallier de la Tour|	White|	NaN	|Inzolia, Italian White|	Sicilia, Sicily & Sardinia, Italy|	12%	|$13|	87|	Kerin O’Keefe|	This 100% Inzolia has aromas of Bosc pear, whi...|
|...|	...|	...|	...|	...|	...|	...|	...|	...|	...|	...|	...|
|323219|	Enrico Serafino 2020 del Comune di Gavi Grifo ...|	Enrico Serafino|	White|	del Comune di Gavi Grifo del Quartaro|	Cortese, Italian White|	Gavi, Piedmont, Italy|	12.5%	|$21|	91|	Kerin O’Keefe|	This savory white opens with enticing scents o...|
|323223|	Grgich Hills 2001 Private Reserve Style Fumé B...|	Grgich Hills|	White	|Private Reserve Style|	Fumé Blanc, Sauvignon Blanc|	Napa Valley, Napa, California, US	|NaN|	$18	|85|	NaN	|Aggressively grassy and tart, with strong flav...|
|323231|	Bott Frères 2017 Tradition Gewurztraminer (Als...|	Bott Frères|	White|	Tradition|	Gewürztraminer, Gewürztraminer|	Alsace, Alsace, France|	13%|	$50|	88|	Anne Krebiehl MW|	Rich honeyed notes of baked apple and rose abo...|
|323233|	Toscolo 2015 Vernaccia di San Gimignano|	Toscolo|	White|	NaN	|Vernaccia, Italian White|	Vernaccia di San Gimignano, Tuscany, Italy	|12.5%	|$11|87	|Kerin O’Keefe|	Aromas of white spring flower, yellow pear and...|
|323234|	Domaine G. Metz 2017 Pinot Blanc (Alsace)|	Domaine G. Metz|	White|	NaN	|Pinot Blanc|	Alsace, Alsace, France	|13%|	$20|	90|	Anne Krebiehl MW|	A tinge of earth clings to the ripe, almost ju...|




### 2.3 추출한 데이터에 대한 탐색적 데이터 분석
1~5점 척도인 경우에는 분포 리뷰 문장의 길이 <br>
연도별, 장소별 등등 <br>
데이터의 부가정보를 바탕으로 데이터를 탐색 ( pandas, matplotlib)

# 여기까지가 중간 과제 점검

## 3. 학습 데이터 구축
## 4. MobileBERT 학습 결과
## 5. 

