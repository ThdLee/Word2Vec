import domain.HiddenNeuron;
import domain.Neuron;
import domain.WordNeuron;
import util.Haffman;
import util.MapCount;

import java.io.*;
import java.util.*;

public class Word2VecLearn {

    private Map<String, Neuron> wordMap = new HashMap<>();
    private MapCount<String> mc;
    /**
     * 训练多少个特征
     */
    private int layerSize = 200;

    /**
     * 上下文窗口大小
     */
    private int window = 5;
    private int negative = 10;

    private double sample = 1e-3;
    private double alpha = 0.025;
    private double startingAlpha = alpha;

    private static final int EXP_TABLE_SIZE = 1000;
    private static final int TABLE_SIZE = (int) 1e8;

    private int trainWordsCount = 0;

    private boolean isCbow = false;
    private boolean isNegative = true;

    private double[] expTable = new double[EXP_TABLE_SIZE];
    private int[] table = null;

    static private final int MAX_EXP = 6;
    private ArrayList<String[]> data = new ArrayList<>();

    public Word2VecLearn(Boolean isCbow, Boolean isNegative, Integer layerSize, Integer window, Double alpha, Double sample, Integer negative) {
        createExpTable();
        if (isCbow != null) this.isCbow = isCbow;
        if (isNegative != null) this.isNegative = isNegative;
        if (layerSize != null) this.layerSize = layerSize;
        if (window != null) this.window = window;
        if (alpha != null) this.alpha = alpha;
        if (sample != null) this.sample = sample;
        if (negative != null) this.negative = negative;
    }

    public Word2VecLearn() {
        createExpTable();
    }

    private void trainModel()  {
        long startTime = System.currentTimeMillis();

        Collections.shuffle(data);

        Long nextRandom = 5l;
        int wordCount = 0;
        int lastWordCount = 0;
        int wordCountActual = 0;

        for (String[] strs : data) {
            if (wordCount - lastWordCount > 10000) {
                System.out.println("alpha:" + alpha + "\tPrigress: "
                        + (int) (wordCountActual / (double) (trainWordsCount + 1) * 100)
                        + "%");
                wordCountActual += wordCount - lastWordCount;
                lastWordCount = wordCount;
                alpha = startingAlpha * (1 - wordCountActual / (double) (trainWordsCount + 1));
                if (alpha < startingAlpha * 0.0001) {
                    alpha = startingAlpha * 0.0001;
                }
            }
            wordCount += strs.length;
            List<WordNeuron> sentence = new ArrayList<WordNeuron>();
            for (int i = 0; i < strs.length; i++) {
                Neuron entry = wordMap.get(strs[i]);
                if (entry == null) continue;

                if (sample > 0) {
                    double ran = (Math.sqrt(entry.freq / (sample * trainWordsCount)) + 1)
                            * (sample * trainWordsCount) / entry.freq;
                    nextRandom = nextRandom * 25214903917L + 11;
                    if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
                        continue;
                    }
                }
                sentence.add((WordNeuron) entry);
            }

            for (int index = 0; index < sentence.size(); index++) {
                nextRandom = nextRandom * 25214903917L + 11;
                if (isCbow) {
                    cbowGram(index, sentence, (int) Math.abs(nextRandom % window), nextRandom);
                } else {
                    skipGram(index, sentence, (int) Math.abs(nextRandom % window), nextRandom);
                }
            }
        }
        long time = (System.currentTimeMillis() - startTime) / 1000;
        System.out.println("Vocab size: " + wordMap.size());
        System.out.println("Words in train file: " + trainWordsCount);
        System.out.println("elapsed time: " + time);
    }

    /**
     * skip-gram
     */
    private void skipGram(int index, List<WordNeuron> sentence, int b, Long nextRandom) {
        WordNeuron word = sentence.get(index);
        int a, c = 0, d = 0;
        for (a = b; a < window * 2 + 1 - b; a++) {
            if (a == window) continue;
            c = index - window + a;
            if (c < 0 || c >= sentence.size()) continue;

            double[] neu1e = new double[layerSize]; // 误差项
            // Hierarchical Softmax
            List<Neuron> neurons = word.neurons;
            WordNeuron we = sentence.get(c);
            for (int i = 0; i < neurons.size(); i++) {
                HiddenNeuron out = (HiddenNeuron) neurons.get(i);
                double f = 0;
                // Propagate hidden -> output
                for (int j = 0; j < layerSize; j++) {
                    f += we.syn0[j] * out.syn1[j];
                }
                if (f <= -MAX_EXP || f >= MAX_EXP) {
                    continue;
                } else {
                    f = (f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2);
                    f = expTable[(int) f];
                }
                // g is the gradient multiplied by the learning rate
                double g = (1 - word.codeArr[i] - f) * alpha;
                // Propagate errors output -> hidden
                for (c = 0; c < layerSize; c++) {
                    neu1e[c] += g * out.syn1[c];
                }
                // Word2VecLearn weights hidden -> output
                for (c = 0; c < layerSize; c++) {
                    out.syn1[c] += g * we.syn0[c];
                }
            }

            if (isNegative) {
                WordNeuron target;
                int label = 0;
                for (d = 0; d < negative + 1; d++) {
                    if (d == 0) {
                        target = word;
                        label = 1;
                    } else {
                        nextRandom = nextRandom * 25214903917L + 11;
                        int i = (int)Math.abs((nextRandom >> 16) % TABLE_SIZE);
                        target = (WordNeuron) wordMap.get(mc.getKey(table[i]));
                        if (target == null) {
                            i = (int) Math.abs(nextRandom % ((wordMap.size() - 1) + 1));
                            target = (WordNeuron) wordMap.get(mc.getKey(table[i]));
                        }
                        if (target == word) continue;
                        label = 0;
                    }
                    double f = 0, g;
                    for (c = 0; c < layerSize; c++) {
                        f += we.syn0[c] * target.syn1neg[c];
                    }
                    if (f > MAX_EXP) g = (label - 1) * alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * alpha;
                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                    for (c = 0; c < layerSize; c++) neu1e[c] += g * target.syn1neg[c];
                    for (c = 0; c < layerSize; c++) target.syn1neg[c] += g * we.syn0[c];
                }
            }

            // Word2VecLearn weights input -> hidden
            for (int j = 0; j < layerSize; j++) {
                we.syn0[j] += neu1e[j];
            }
        }
    }

    private void cbowGram(int index, List<WordNeuron> sentence, int b, Long nextRandom) {
        WordNeuron word = sentence.get(index);
        int a, c = 0, d = 0;

        List<Neuron> neurons = word.neurons;
        double[] neu1e = new double[layerSize]; // 误差项
        double[] neu1 = new double[layerSize];  // 误差项
        WordNeuron last_word;

        for (a = b; a < window * 2 + 1 - b; a++) {
            if (a != window) {
                c = index - window + a;
                if (c < 0 || c >= sentence.size()) continue;
                last_word = sentence.get(c);
                if (last_word == null) continue;
                for (c = 0; c < layerSize; c++) {
                    neu1[c] += last_word.syn0[c];
                }
            }
        }

        for (d = 0; d < neurons.size(); d++) {
            HiddenNeuron out = (HiddenNeuron) neurons.get(d);
            double f = 0;
            // Propagate hidden -> output
            for (c = 0; c < layerSize; c++) {
                f += neu1[c] * out.syn1[c];
            }
            if (f <= -MAX_EXP || f >= MAX_EXP) continue;
            f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP/ 2))];
            // 'g' is the gradient multiplied by the learning rate
            double g = f * (1 - f) * (word.codeArr[d] - f) * alpha;
            // Propagate errors output -> hidden
            for (c = 0; c < layerSize; c++) {
                neu1e[c] += g * out.syn1[c];
            }
            // Word2VecLearn weights hidden -> output
            for (c = 0; c < layerSize; c++) {
                out.syn1[c] += g * neu1[c];
            }
        }

        if (isNegative) {
            WordNeuron target;
            int label = 0;
            for (d = 0; d < negative + 1; d++) {
                if (d == 0) {
                    target = word;
                    label = 1;
                } else {
                    nextRandom = nextRandom * 25214903917L + 11;
                    int i = (int)Math.abs((nextRandom >> 16) % TABLE_SIZE);
                    target = (WordNeuron) wordMap.get(mc.getKey(table[i]));
                    if (target == null) {
                        i = (int)Math.abs(nextRandom % ((wordMap.size() - 1) + 1));
                        target = (WordNeuron) wordMap.get(mc.getKey(table[i]));
                    }
                    if (target == word) continue;
                    label = 0;
                }
                double f = 0, g;
                for (c = 0; c < layerSize; c++) {
                    f += neu1[c] * target.syn1neg[c];
                }
                if (f > MAX_EXP) g = (label - 1) * alpha;
                else if (f < -MAX_EXP) g = (label - 0) * alpha;
                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                for (c = 0; c < layerSize; c++) neu1e[c] += g * target.syn1neg[c];
                for (c = 0; c < layerSize; c++) target.syn1neg[c] += g * neu1[c];
            }
        }
        for (a = b; a < window * 2 + 1 - b; a++) {
            if (a != window) {
                c = index - window + a;
                if (c < 0 || c >= sentence.size()) continue;
                last_word = sentence.get(c);
                if (last_word == null) continue;
                for (c = 0; c < layerSize; c++) {
                    last_word.syn0[c] += neu1e[c];
                }
            }
        }
    }

    private void readVocab(File file) throws IOException {
        mc = new MapCount<>();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {

            String temp = null;
            while ((temp = br.readLine()) != null) {
                String[] split = temp.split("\t+| +");
                trainWordsCount += split.length;
                for (String string : split) {
                    mc.add(string);
                }
                data.add(split);
            }
        }
        for (Map.Entry<String, Integer> element : mc.get().entrySet()) {
            wordMap.put(element.getKey(), new WordNeuron(element.getKey(), (double) element.getValue() / mc.size(), layerSize));
        }
    }

    private void readVocab(File[] files) throws IOException {
        mc = new MapCount<>();
        for (File file : files) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {

                String temp = null;
                while ((temp = br.readLine()) != null) {
                    String[] split = temp.split("\t+| +");
                    trainWordsCount += split.length;
                    for (String string : split) {
                        mc.add(string);
                    }
                    data.add(split);
                }
            }
        }

        for (Map.Entry<String, Integer> element : mc.get().entrySet()) {
            wordMap.put(element.getKey(), new WordNeuron(element.getKey(), (double) element.getValue() / mc.size(), layerSize));
        }
    }

    private void createExpTable() {
        for (int i = 0; i < EXP_TABLE_SIZE; i++) {
            expTable[i] = Math.exp((i / (double) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
            expTable[i] = expTable[i] / (expTable[i] + 1);
        }
    }

    private void initializeUnigramTable() {
        table = new int[TABLE_SIZE];
        long trainWordsPow = 0;
        double power = 0.75;
        for (Integer integer : mc.get().values()) { trainWordsPow += Math.pow(integer, power); }
        Iterator<Integer> nodeIter = mc.get().values().iterator();
        Integer last = nodeIter.next();
        double d1 = Math.pow(last, power) / trainWordsPow;
        int i = 0;
        for (int a = 0; a < TABLE_SIZE; a++) {
            table[a] = i;
            if (a / (double) TABLE_SIZE > d1) {
                i++;
                Integer next = nodeIter.hasNext() ? nodeIter.next() : last;
                d1 += Math.pow(next, power) / trainWordsPow;
                last = next;
            }
        }
    }

    public void learnFile(File file, int epoch) throws IOException {
        readVocab(file);
        new Haffman(layerSize).make(wordMap.values());
        if (isNegative) initializeUnigramTable();
        // 查找每个神经元
        for (Neuron neuron : wordMap.values()) {
            ((WordNeuron) neuron).makeNeurons();
        }
        long startTime = System.currentTimeMillis();
        for (int i = 1; i <= epoch; i++) {
            System.out.println("Epoch " + i);
            trainModel();
        }
        long time = (System.currentTimeMillis() - startTime) / 1000;
        System.out.println("time-consuming " + time);
    }

    public void learnFile(File[] files, int epoch) throws IOException {
        readVocab(files);
        new Haffman(layerSize).make(wordMap.values());
        if (isNegative) initializeUnigramTable();
        // 查找每个神经元
        for (Neuron neuron : wordMap.values()) {
            ((WordNeuron) neuron).makeNeurons();
        }
        long startTime = System.currentTimeMillis();
        for (int i = 1; i <= epoch; i++) {
            System.out.println("Epoch " + i);
            trainModel();
        }
        long time = (System.currentTimeMillis() - startTime) / 1000;
        System.out.println("time-consuming " + time);
    }

    // 保存模型
    public void saveModel(File file) {
        try (DataOutputStream dataOutputStream = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)))) {
            dataOutputStream.writeInt(wordMap.size());
            dataOutputStream.writeInt(layerSize);
            double[] syn0 = null;
            for (Map.Entry<String, Neuron> element : wordMap.entrySet()) {
                dataOutputStream.writeUTF(element.getKey());
                syn0 = ((WordNeuron) element.getValue()).syn0;
                for (double d : syn0) {
                    dataOutputStream.writeFloat(((Double) d).floatValue());
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public Map<String, Neuron> getWord2VecModel() {

        return wordMap;
    }

    public int getLayerSize() {
        return layerSize;
    }

    public void setLayerSize(int layerSize) {
        this.layerSize = layerSize;
    }

    public int getWindow() {
        return window;
    }

    public void setWindow(int window) {
        this.window = window;
    }

    public double getSample() {
        return sample;
    }

    public void setSample(double sample) {
        this.sample = sample;
    }

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
        this.startingAlpha = alpha;
    }

    public boolean getIsCbow() {
        return isCbow;
    }

    public void setIsCbow(boolean cbow) {
        isCbow = cbow;
    }

    public boolean getIsNegative() {
        return isNegative;
    }

    public void setIsNegative(boolean isNegative) {
        this.isNegative = isNegative;
    }

}
