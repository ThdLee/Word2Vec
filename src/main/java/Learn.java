import domain.HiddenNeuron;
import domain.Neuron;
import domain.WordNeuron;
import util.Haffman;
import util.MapCount;

import java.io.*;
import java.util.*;

public class Learn {

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

    public Learn(Boolean isCbow, Boolean isNegative, Integer layerSize, Integer window, Double alpha, Double sample, Integer negative) {
        createExpTable();
        if (isCbow != null) this.isCbow = isCbow;
        if (isNegative != null) this.isNegative = isNegative;
        if (layerSize != null) this.layerSize = layerSize;
        if (window != null) this.window = window;
        if (alpha != null) this.alpha = alpha;
        if (sample != null) this.sample = sample;
        if (negative != null) this.negative = negative;
    }

    public Learn() {
        createExpTable();
    }

    private void trainModel(File file) throws IOException {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
            String temp = null;
            Long nextRandom = 5l;
            int wordCount = 0;
            int lastWordCount = 0;
            int wordCountActual = 0;
            while ((temp = br.readLine()) != null) {
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
                String[] strs = temp.split(" +|\t+");
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
                        cbowGram(index, sentence, (int) Long.remainderUnsigned(nextRandom, window), nextRandom);
                    } else {
                        skipGram(index, sentence, (int) Long.remainderUnsigned(nextRandom, window), nextRandom);
                    }
                }
            }
            System.out.println("Vocab size: " + wordMap.size());
            System.out.println("Words in train file: " + trainWordsCount);
            System.out.println("success train over!");
        }
    }

    /**
     * skip-gram
     */
    private void skipGram(int index, List<WordNeuron> sentence, int b, Long nextRandom) {
        WordNeuron word = sentence.get(index);
        int a, c = 0, d = 0;
        for (a = b; a < window * 2 + 1 - b; a++) {
            if (a == window) continue;
            c = index - window + 1;
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
                // Learn weights hidden -> output
                for (c = 0; c < layerSize; c++) {
                    out.syn1[c] += g * we.syn0[c];
                }
            }
            WordNeuron target;
            int label = 0;
            if (isNegative) {
                for (d = 0; d < negative + 1; d++) {
                    if (d == 0) {
                        target = word;
                        label = 1;
                    } else {
                        nextRandom = nextRandom * 25214903917L + 11;
                        int i = (int)((nextRandom >> 16) % TABLE_SIZE);
                        target = (WordNeuron) wordMap.get(mc.getKey(i));
                        if (target == null) {
                            i = (int)Long.remainderUnsigned(nextRandom, (wordMap.size() - 1) + 1);
                            target = (WordNeuron) wordMap.get(mc.getKey(i));
                        }
                        if (target == word) continue;
                        label = 0;
                    }
                    double f = 0, g;
                    for (c = 0; c < layerSize; c++) {
                        f += neu1e[c] * target.syn1neg[c];
                    }
                    if (f > MAX_EXP) g = (label - 1) * alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * alpha;
                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                    for (c = 0; c < layerSize; c++) neu1e[c] += g * target.syn1neg[c];
                    for (c = 0; c < layerSize; c++) target.syn1neg[c] += g * neu1e[c];
                }
            }

            // Learn weights input -> hidden
            for (int j = 0; j < layerSize; j++) {
                we.syn0[j] += neu1e[j];
            }
        }
    }

    private void cbowGram(int index, List<WordNeuron> sentence, long b, Long nextRandom) {
        WordNeuron word = sentence.get(index);
        int a, c = 0, d = 0;

        List<Neuron> neurons = word.neurons;
        double[] neu1e = new double[layerSize]; // 误差项
        double[] neu1 = new double[layerSize];  // 误差项
        WordNeuron last_word;

        for (a = (int)b; a < window * 2 + 1 - b; a++) {
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
            // Learn weights hidden -> output
            for (c = 0; c < layerSize; c++) {
                out.syn1[c] += g * neu1[c];
            }
        }

        WordNeuron target;
        int label = 0;
        if (isNegative) {
            for (d = 0; d < negative + 1; d++) {
                if (d == 0) {
                    target = word;
                    label = 1;
                } else {
                    nextRandom = nextRandom * 25214903917L + 11;
                    int i = (int)((nextRandom >> 16) % TABLE_SIZE);
                    target = (WordNeuron) wordMap.get(mc.getKey(i));
                    if (target == null) {
                        i = (int)Long.remainderUnsigned(nextRandom, (wordMap.size() - 1) + 1);
                        target = (WordNeuron) wordMap.get(mc.getKey(i));
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
        for (a = (int)b; a < window * 2 + 1 - b; a++) {
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
            }
        }
        for (Map.Entry<String, Integer> element : mc.get().entrySet()) {
            wordMap.put(element.getKey(), new WordNeuron(element.getKey(), (double) element.getValue() / mc.size(), layerSize));
        }
    }

    private void readVocabWithSupervised(File[] files) throws IOException {
        for (int category = 0; category < files.length; category++) {
            // 对多个文件学习
            mc = new MapCount<>();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(files[category])))) {
                String temp = null;
                while ((temp = br.readLine()) != null) {
                    String[] split = temp.split(" ");
                    trainWordsCount += split.length;
                    for (String string : split) {
                        mc.add(string);
                    }
                }
            }
            for (Map.Entry<String, Integer> element : mc.get().entrySet()) {
                double tarFreq = (double) element.getValue() / mc.size();
                if (wordMap.get(element.getKey()) != null) {
                    double srcFreq = wordMap.get(element.getKey()).freq;
                    if (srcFreq >= tarFreq) continue;
                    Neuron wordNeuron = wordMap.get(element.getKey());
                    wordNeuron.category = category;
                    wordNeuron.freq = tarFreq;
                } else {
                    wordMap.put(element.getKey(), new WordNeuron(element.getKey(), tarFreq, category, layerSize));
                }
            }
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

    public void learnFile(File file) throws IOException {
        readVocab(file);
        new Haffman(layerSize).make(wordMap.values());
        if (isNegative) initializeUnigramTable();
        // 查找每个神经元
        for (Neuron neuron : wordMap.values()) {
            ((WordNeuron) neuron).makeNeurons();
        }

        trainModel(file);
    }

    public void learnFile(File summaryFile, File[] classifiedFiles) throws IOException {
        readVocabWithSupervised(classifiedFiles);
        new Haffman(layerSize).make(wordMap.values());
        if (isNegative) initializeUnigramTable();
        // 查找每个神经元
        for (Neuron neuron : wordMap.values()) {
            ((WordNeuron) neuron).makeNeurons();
        }
        trainModel(summaryFile);
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

    public static void main(String[] args) throws IOException {
        Learn learn = new Learn();
        long start = System.currentTimeMillis();
        learn.learnFile(new File("library/xh.txt"));
        System.out.println("use time " + (System.currentTimeMillis() - start));
        learn.saveModel(new File("library/javaVector"));
    }
}
