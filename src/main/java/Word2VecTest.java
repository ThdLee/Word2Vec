
import domain.WordEntry;

import java.io.File;
import java.io.IOException;




public class Word2VecTest {
    private static final File sportCorpusFile = new File("corpus/result.txt");

    public static void main(String[] args) throws IOException {
        String path = "corpus/sport/";
        String[] filenames = {"train-pos.txt", "train-neg.txt", "test-pos.txt", "test-neg.txt"};
        File[] files = new File[filenames.length];
        for (int i = 0; i < filenames.length; i++) {
            files[i] = new File(path + filenames[i]);
        }


        Word2VecLearn lean = new Word2VecLearn() ;

        lean.learnFile(files, 1) ;

        lean.saveModel(new File("model/vector.mod")) ;



        //加载测试

        Word2Vec w2v = new Word2Vec() ;

        w2v.loadJavaModel("model/vector.mod") ;

        System.out.println("queen - woman + man = ");
        for (WordEntry word : w2v.analogy("queen", "woman", "man")) {
            System.out.println(word.name + " " + word.score);
        }
    }


}
