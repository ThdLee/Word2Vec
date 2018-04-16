# Word2Vec

Vocabulary Size : 151940
Words in train file: 7345315

## Training Time (second)

||no negative sampling| negative sampling|
|-----|:-----:|:-----:|
|cbow|51|173|
|skip-gram|147|491|

## Distance (top10)
### 韩国
```
泰国 0.87579507
澳大利亚 0.87488353
印尼 0.87280625
马来西亚 0.8718894
丹麦 0.86428016
德国 0.8620918
加拿大 0.86166114
新加坡 0.8555199
荷兰 0.8549148
意大利 0.85362816
```
## Analogy (top20)
### 广东 - 广州 + 南京
```
河北 0.91932815
安徽 0.9100935
河南 0.90377665
江西 0.9023698
吉林 0.8972926
云南 0.88759595
黑龙江 0.8874138
浙江 0.8808151
陕西 0.87843263
湖南 0.87814873
丹东 0.87484443
山西 0.8736979
湖北 0.8702805
襄樊市 0.8692638
福建 0.8674251
英山县 0.8670151
上饶 0.86647516
襄樊 0.8659118
常州市 0.8645202
四川省 0.8621572
```

### 德国 - 柏林 + 伦敦
```
英国 0.8039676
库克 0.8019326
意大利 0.7980183
布莱尔 0.79549104
俄罗斯 0.7933194
泰国 0.7899806
普里马科夫 0.78864676
马德里 0.7879681
主席国 0.78405714
韩国 0.7833164
比利时 0.7825046
乌克兰 0.7806997
理查森 0.7804407
卢森堡 0.7797551
加拿大 0.77851707
内塔尼亚胡 0.77737486
丹麦 0.7747326
苏哈托 0.7737011
菲律宾 0.7727813
萨哈夫 0.7726762
```

# Doc2Vec

Vocabulary Size : 100674
Words in train file: 11711200

## Dataset

The dataset consist of 100,000 movie reviews taken from IMDB. There are 25,000 labeled training instances and 25,000 labeled test instances. The dataset can be downloaded at 	 http://ai.Stanford.edu/amaas/data/sentiment/index.html

## Result

50,000 labeled instances were be learned to generate word vectors and paragraph vectors. Then we use Word2Vec for sentiment analysis by attempting to classify the Cornell IMDB movie review corpus.

|  Model  | Accuracy |
| :-----: | :------: |
|   DM    | 0.74484  |
|  DBOW   | 0.71484  |
| DM+DBOW | 0.85368  |




​			
​		
​	