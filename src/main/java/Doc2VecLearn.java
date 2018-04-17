import domain.HiddenNeuron;
import domain.Neuron;
import domain.WordNeuron;
import javafx.util.Pair;
import util.Haffman;
import util.MapCount;

import java.io.*;
import java.util.*;

public class Doc2VecLearn {
    public enum Model {
        DBOW,
        DM,
        BOTH
    }

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

    private Model model = Model.DM;
    private boolean isNegative = true;

    private double[] expTable = new double[EXP_TABLE_SIZE];
    private int[] table = null;

    static private final int MAX_EXP = 6;

    private Map<Integer, float[]> docMap = new HashMap<>();
    private ArrayList<Pair<Integer, String[]>> docData = new ArrayList<>();

    public Doc2VecLearn(Model model, Boolean isNegative, Integer layerSize, Integer window, Double alpha, Double sample, Integer negative) {
        createExpTable();
        if (model != null) this.model = model;
        if (isNegative != null) this.isNegative = isNegative;
        if (layerSize != null) this.layerSize = layerSize;
        if (window != null) this.window = window;
        if (alpha != null) this.alpha = alpha;
        if (sample != null) this.sample = sample;
        if (negative != null) this.negative = negative;
    }

    public Doc2VecLearn() {
        createExpTable();
    }

    private void trainModel() {
        long startTime = System.currentTimeMillis();

        Collections.shuffle(docData);

        Long nextRandom = 5l;
        int wordCount = 0;
        int lastWordCount = 0;
        int wordCountActual = 0;

        for (Pair<Integer, String[]> pair : docData) {
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
            String[] strs = pair.getValue();
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
                if (model == Model.DM) {
                    dm(index, pair.getKey(), sentence, (int) Math.abs(nextRandom % window), nextRandom);
                } else if (model == Model.DBOW) {
                    dbow(index, pair.getKey(), sentence, (int) Math.abs(nextRandom % window), nextRandom);
                } else {
                    dm(index, pair.getKey(), sentence, (int) Math.abs(nextRandom % window), nextRandom);
                    dbow(index, pair.getKey(), sentence, (int) Math.abs(nextRandom % window), nextRandom);
                }
            }
        }
        long time = (System.currentTimeMillis() - startTime) / 1000;
        System.out.println("Vocab size: " + wordMap.size());
        System.out.println("Words in train file: " + trainWordsCount);
        System.out.println("elapsed time: " + time);

    }

    /**
     * PV-DBOW
     */
    private void dbow(int index, int sentNo, List<WordNeuron> sentence, int b, Long nextRandom) {
        WordNeuron word = sentence.get(index);
        int s = model == Model.BOTH ? layerSize : 0;
        float[] senVec = docMap.get(sentNo);
        int a, c = 0, d = 0;
        for (a = b; a < window * 2 + 1 - b; a++) {
            if (a == window) continue;
            c = index - window + a;
            if (c < 0 || c >= sentence.size()) continue;

            double[] neu1e = new double[layerSize]; // 误差项
            // Hierarchical Softmax
            List<Neuron> neurons = word.neurons;
            for (int i = 0; i < neurons.size(); i++) {
                HiddenNeuron out = (HiddenNeuron) neurons.get(i);
                double f = 0;
                // Propagate hidden -> output
                for (int j = 0; j < layerSize; j++) {
                    f += senVec[j+s] * out.syn1[j];
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
                    out.syn1[c] += g * senVec[c+s];
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
                        f += senVec[c+s] * target.syn1neg[c];
                    }
                    if (f > MAX_EXP) g = (label - 1) * alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * alpha;
                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                    for (c = 0; c < layerSize; c++) neu1e[c] += g * target.syn1neg[c];
                    for (c = 0; c < layerSize; c++) target.syn1neg[c] += g * senVec[c+s];
                }
            }

            // Word2VecLearn weights input -> hidden
            for (int j = 0; j < layerSize; j++) {
                senVec[j+s] += neu1e[j];
            }
        }
    }

    /**
     * PV-DM
     */
    private void dm(int index, int sentNo, List<WordNeuron> sentence, int b, Long nextRandom) {
        WordNeuron word = sentence.get(index);
        int a, c = 0, d = 0;

        float[] senVec = docMap.get(sentNo);

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
        for (c = 0; c < layerSize; c++) {
            neu1[c] += senVec[c];
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
        for (c = 0; c < layerSize; c++) {
            senVec[c] += neu1e[c];
        }
    }



    private void createExpTable() {
        for (int i = 0; i < EXP_TABLE_SIZE; i++) {
            expTable[i] = Math.exp((i / (double) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
            expTable[i] = expTable[i] / (expTable[i] + 1);
        }
    }

    private void initializeDocVec(File file) throws IOException {
        mc = new MapCount<>();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(file)))) {
            String temp = null;

            int sent_no = 0;
            while ((temp = br.readLine()) != null) {
                String[] split = temp.split(" ");
                docData.add(new Pair<>(sent_no, split));
                trainWordsCount += split.length;
                for (String string : split) {
                    mc.add(string);
                }

                float[] vector;
                if (model == Model.BOTH) vector = new float[layerSize * 2];
                else vector = new float[layerSize];

                Random random = new Random();

                for (int i = 0; i < vector.length; i++)
                    vector[i] = (float) ((random.nextDouble() - 0.5) / layerSize);

                docMap.put(sent_no, vector);
                sent_no++;
            }
        }
        for (Map.Entry<String, Integer> element : mc.get().entrySet()) {
            wordMap.put(element.getKey(), new WordNeuron(element.getKey(), (double) element.getValue() / mc.size(), layerSize));
        }
    }

    private void initializeDocVec(File[] files) throws IOException {
        mc = new MapCount<>();
        int sent_no = 0;
        for (File file : files) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(
                    new FileInputStream(file)))) {
                String temp = null;

                while ((temp = br.readLine()) != null) {
                    String[] split = temp.split(" ");
                    docData.add(new Pair<>(sent_no, split));
                    trainWordsCount += split.length;
                    for (String string : split) {
                        mc.add(string);
                    }
                    float[] vector;
                    if (model == Model.BOTH) vector = new float[layerSize * 2];
                    else vector = new float[layerSize];
                    Random random = new Random();

                    for (int i = 0; i < vector.length; i++)
                        vector[i] = (float) ((random.nextDouble() - 0.5) / layerSize);

                    docMap.put(sent_no, vector);
                    sent_no++;
                }
            }
            System.out.println(file.toString() + "is done. " + docData.size() + " lines");
        }
        for (Map.Entry<String, Integer> element : mc.get().entrySet()) {
            wordMap.put(element.getKey(), new WordNeuron(element.getKey(), (double) element.getValue() / mc.size(), layerSize));
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
        initializeDocVec(file);
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
        initializeDocVec(files);
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

    public void saveDocVec(String path) {
        try (FileWriter fw = new FileWriter(path)) {
            for (int i = 0; i < docMap.size(); i++){
                StringBuilder sb = new StringBuilder();
                sb.append(i);
                for (float f : docMap.get(i)) {
                    sb.append(',').append(f);
                }
                fw.write(sb.append('\n').toString());
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

    public Model getModel() {
        return model;
    }

    public void setModel(Model model) {
        this.model = model;
    }

    public boolean getIsNegative() {
        return isNegative;
    }

    public void setIsNegative(boolean isNegative) {
        this.isNegative = isNegative;
    }

}
