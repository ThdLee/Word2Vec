
import domain.WordEntry;

import java.io.File;
import java.io.IOException;




public class Word2VecTest {
    private static final File sportCorpusFile = new File("corpus/result.txt");

    public static void main(String[] args) throws IOException {
//        File file = new File("corpus/sport/swresult_withoutnature.txt");
//
        //进行分词训练

//        Word2VecLearn lean = new Word2VecLearn() ;
//
//        lean.learnFile(file) ;
//
//        lean.saveModel(new File("model/vector.mod")) ;



        //加载测试

        Word2Vec w2v = new Word2Vec() ;

        w2v.loadJavaModel("model/vector.mod") ;

        System.out.println("广东 - 广州 + 南京 = ");
        for (WordEntry word : w2v.analogy("广州", "广东", "南京")) {
            System.out.println(word.name + " " + word.score);
        }
    }


}
