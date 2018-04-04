package util;

import java.util.*;

public class WordKmeans {



    private HashMap<String, float[]> wordMap = null;

    private int iter;

    private Classes[] cArray = null;

    public WordKmeans(HashMap<String, float[]> wordMap, int clcn, int iter) {
        this.wordMap = wordMap;
        this.iter = iter;
        cArray = new Classes[clcn];
    }

    public Classes[] explain() {
        // 取前 clcn 个点
        Iterator<Map.Entry<String, float[]>> iterator = wordMap.entrySet().iterator();
        for (int i = 0; i < cArray.length; i++) {
            Map.Entry<String, float[]> next = iterator.next();
            cArray[i] = new Classes(i, next.getValue());
        }

        for (int i = 0; i < iter; i++) {
            for (Classes classes : cArray) {
                classes.clean();
            }

            iterator = wordMap.entrySet().iterator();
            while (iterator.hasNext()) {
                Map.Entry<String, float[]> next = iterator.next();
                double miniScore = Double.MAX_VALUE;
                double tempScore;
                int classesId = 0;
                for (Classes classes : cArray) {
                    tempScore = classes.distance(next.getValue());
                    if (miniScore > tempScore) {
                        miniScore = tempScore;
                        classesId = classes.id;
                    }
                }
                cArray[classesId].putValue(next.getKey(), miniScore);
            }

            for (Classes classes : cArray) {
                classes.updateCenter(wordMap);
            }
            System.out.println("iter " + i + " ok!");
        }
        return cArray;
    }

    public static class Classes {
        private int id;

        private float[] center;

        public Classes(int id, float[] center) {
            this.id = id;
            this.center = center.clone();
        }

        Map<String, Double> values = new HashMap<>();

        public double distance(float[] value) {
            double sum = 0;
            for (int i = 0; i < value.length; i++) {
                sum += (center[i] - value[i]) * (center[i] - value[i]);
            }
            return sum;
        }

        public void putValue(String word, double score) {
            values.put(word, score);
        }

        // 重新计算中心点
        public void updateCenter(HashMap<String, float[]> wordMap) {
            for (int i = 0; i < center.length; i++) {
                center[i] = 0;
            }
            float[] value = null;
            for (String keyWord : values.keySet()) {
                value = wordMap.get(keyWord);
                for (int i = 0; i < value.length; i++) {
                    center[i] += value[i];
                }
            }
            for (int i = 0; i < center.length; i++) {
                center[i] = center[i] / values.size();
            }
        }

        // 清空历史结果
        public void clean() {
            values.clear();
        }

        // 取得每个类别的前n个结果
        public List<Map.Entry<String, Double>> getTop(int n) {
            List<Map.Entry<String, Double>> arrayList = new ArrayList<>(values.entrySet());
            Collections.sort(arrayList, new Comparator<Map.Entry<String, Double>>() {
                @Override
                public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {
                    return o1.getValue() > o2.getValue() ? 1 : -1;
                }
            });
            int min = Math.min(n, arrayList.size() - 1);
            if (min <= 1) return Collections.emptyList();
            return arrayList.subList(0, min);
        }
    }

}
