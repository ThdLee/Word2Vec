package domain;

public class HiddenNeuron extends Neuron {
    public double[] syn1;

    public HiddenNeuron(int layerSize) {
        syn1 = new double[layerSize];
    }
}
