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
화이트 와인 데이터는 95424건이다.<br/>

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
평점(rating)은 80점부터 100점까지 구성되어있다.<br/>
![rt1](https://github.com/qkrtnals/Wine_Review/assets/79901070/bfb3dc21-79d9-43a8-bc54-54a4d99505e4) <br/>
![rt2](https://github.com/qkrtnals/Wine_Review/assets/79901070/e07189a6-c85a-40ad-bda2-4cce82b3f3ad)


화이트 와인 중 평점이 높은 상위 5개의 리뷰 보기 <br/>
![상위5개리뷰](https://github.com/qkrtnals/-/assets/79901070/94e66106-1c73-495f-9bed-9fee7c136cd0)

|wine|review|
|--|--|
|Ramey 2018 Rochioli Vineyard Chardonnay (Russian River Valley)| This wine is a testament to an outstanding site in the hands of an outstanding producer. It is a celebration of fresh Meyer lemon and Gravenstein apple woven against an elegant structure and quenching acidity|
|Château Haut-Brion 2014 Pessac-Léognan | Full of ripe fruit, opulent and concentrated, this is a fabulous and impressive wine. It has a beautiful line of acidity balanced with ripe fruits. The wood aging is subtle, just a hint of smokiness and toast. This is one of those wines, from a great white wine vintage, that will age many years. Drink from 2024.|
|Ramey 2018 Hyde Vineyard Chardonnay (Carneros) | This is a stunner from blocks planted in 1997 and 1999. Beautiful aromas of toasted hazelnut and stone lead to a palate that shows undeniable minerality paired with lasting acidity. It shows freshness and energy throughout. Drink 2026–2032. |
|Failla 2010 Estate Vineyard Chardonnay (Sonoma Coast) | Shows classic, full-throttle notes of tropical and citrus fruits, pears and sweet green apples, combined with strong minerality and complex layers of buttered toast, honey and creamy lees. The description alone hardly does justice to the wine's beauty. The acidity is perfect, the oak deftly applied, the finish long and completely satisfying. Winemaker Ehren Jordan suggests pairing it with simple fare like roast chicken and salted fingerling potatoes.|
|Ramey 2017 Rochioli Vineyard Chardonnay (Russian River Valley) | Only the third vintage by this producer from a spectacular site, this elegant wine is brimming in tension and length. Delicate layers of Meyer lemon, Gravenstein apple and pear show an abundance on the palate, seasoned in salty oak, nutmeg and a hint of cardamom.|



연도별, 장소별 등등 <br>

데이터의 부가정보를 바탕으로 데이터를 탐색 ( pandas, matplotlib)

## 3. 학습 데이터 구축
# 3.1 긍부정 분류
평점이 80점부터 100점까지 분포하는데 92.5점 이상은 대부분 긍정적인 표현이 많은 반면  85점 이하의 리뷰는 긍정적인 표현이 없고, 부정적표현이 두드러지지 않고 무미건조한 설명으로 구성된다.

92.5점 이상을 긍정으로 분류하고
85점 이하를 부정으로 분류하겠다.

그래프를 그려보면 전체 9만건의 데이터에서 긍부정으로 분류된 데이터의 수는 26756건이며 
여기서 긍정은 8435건, 부정은 18321건이다.
![11](https://github.com/qkrtnals/Wine_Review/assets/79901070/62de0669-dc2a-4fd0-b4e2-43cf91d63926)
![22](https://github.com/qkrtnals/Wine_Review/assets/79901070/6b14a8d7-2c9e-4d21-8829-d747e8fd78d3)


92.5점 이상을 1, 85점 이하를 0으로 분류한 'sentiment(긍부정)' 칼럼을 추가해보았다.
|-| wine |	review | sentiment|
|-|--|--|--|
|4|	Tenuta San Francesco 2007 Tramonti White (Campania)|	This intriguing blend of Falanghina, Biancolella and Pepella (three relatively unknown indigenous grapes) has a candy...|	0|
|46|	Bellora 2016 Soave	|Fruity aromas of tropical fruit and white peach carry over to the simple palate along with the barest hint of bit...|0|
|60|	Roth 2011 Sauvignon Blanc (Alexander Valley)|	Off-dry to sweetish, this has honey, orange, apricot, lime and vanilla flavors. It shows Sauvignon's clean, brisk aci...|	0|
|...|	...	|...	|...	|
|323102|	Domaine Laporte 2016 Le Rochoy (Sancerre)|Roger Voss	From a sloping flinty vineyard, this is an impressive wine. It is powered by its tight tense texture to give...|	1|
|323153|	William Fèvre 2007 Bougros Grand Cru (Chablis)|A big, burly Chablis, beautifully ripe, exhibiting powerful fruits overlaying a core of citrus and apple skin texture...|1|
|323196|	Ramey 2017 Chardonnay (Russian River Valley)|	This is such a fine appellation wine, punching well above its price point in quality and memorability, sourced from such stellar sites as Westside Farms...|	1|



,
## 4. MobileBERT 학습 결과
## 5. 

