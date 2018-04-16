
import domain.Neuron;

import java.io.*;
import java.util.Map;


public class Doc2VecTest {

    public static void main(String[] args) throws IOException {
        String path = "corpus/sport/";
        String[] filenames = {"train-pos.txt", "train-neg.txt", "test-pos.txt", "test-neg.txt"};
        File[] files = new File[filenames.length];
        for (int i = 0; i < filenames.length; i++) {
            files[i] = new File(path + filenames[i]);
        }

        //进行分词训练

        Doc2VecLearn learn = new Doc2VecLearn() ;

        learn.learnFile(files, 1) ;

        learn.saveDocVec("vector.csv");

    }

}