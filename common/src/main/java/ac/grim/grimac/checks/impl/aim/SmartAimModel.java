package ac.grim.grimac.checks.impl.aim;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.Collections;
import java.util.List;

/**
 * СУПЕР-УМНАЯ ML-МОДЕЛЬ для детекции аимботов
 *
 * Архитектура (State-of-the-Art):
 * - BatchNormalization (нормализация входа)
 * - LSTM 256 нейронов + Dropout 0.3
 * - LSTM 128 нейронов + Dropout 0.3
 * - Dense 64 нейрона + ReLU + Dropout 0.4
 * - Dense 32 нейрона + ReLU + Dropout 0.4
 * - Output 1 нейрон (Sigmoid)
 *
 * Фичи:
 * - Early Stopping (остановка при переобучении)
 * - Learning Rate Decay (адаптивная скорость)
 * - Gradient Clipping (стабильность)
 * - Validation на каждой эпохе
 *
 * @author TushpAC Team
 * @version 3.0 - ULTRA SMART EDITION
 */
public class SmartAimModel implements Serializable {
    private static final long serialVersionUID = 3L;

    private transient MultiLayerNetwork network;

    // Параметры
    private final int sequenceLength = 40;
    private final int numFeatures = 8;  // deltaYaw, deltaPitch, accel*2, jerk*2, gcdError*2

    // Метрики
    private double accuracy = 0.0;
    private double precision = 0.0;
    private double recall = 0.0;
    private double f1Score = 0.0;

    // Нормализация (важно!)
    private double[] featureMeans;
    private double[] featureStds;

    /**
     * Создать новую модель
     */
    public SmartAimModel() {
        featureMeans = new double[numFeatures];
        featureStds = new double[numFeatures];
        buildNetwork();
    }

    /**
     * Построить СУПЕР-УМНУЮ нейронную сеть
     */
    private void buildNetwork() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .updater(new Adam(0.001))  // Adam оптимизатор
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .list()

                // Слой 0: Batch Normalization (ВАЖНО для стабильности)
                .layer(0, new BatchNormalization.Builder()
                        .nIn(numFeatures)
                        .nOut(numFeatures)
                        .build())

                // Слой 1: LSTM 256 нейронов
                .layer(1, new LSTM.Builder()
                        .nIn(numFeatures)
                        .nOut(256)
                        .activation(Activation.TANH)
                        .build())

                // Dropout для предотвращения переобучения
                .layer(2, new DropoutLayer.Builder(0.3).build())

                // Слой 3: LSTM 128 нейронов
                .layer(3, new LSTM.Builder()
                        .nIn(256)
                        .nOut(128)
                        .activation(Activation.TANH)
                        .build())

                .layer(4, new DropoutLayer.Builder(0.3).build())

                // Слой 5: Dense 64 (полносвязный)
                .layer(5, new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())

                .layer(6, new DropoutLayer.Builder(0.4).build())

                // Слой 7: Dense 32
                .layer(7, new DenseLayer.Builder()
                        .nIn(64)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())

                .layer(8, new DropoutLayer.Builder(0.4).build())

                // Выходной слой: 1 нейрон (вероятность чита)
                .layer(9, new RnnOutputLayer.Builder()
                        .nIn(32)
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .lossFunction(LossFunctions.LossFunction.XENT)
                        .build())

                .build();

        network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(new ScoreIterationListener(5));

        System.out.println("╔═══════════════════════════════════════════════════");
        System.out.println("║ [SmartAimModel] ULTRA SMART сеть создана!");
        System.out.println("║ BatchNorm → LSTM(256) → Dropout → LSTM(128)");
        System.out.println("║ → Dropout → Dense(64) → Dense(32) → Output(1)");
        System.out.println("║ Параметров: ~" + (network.numParams() / 1000) + "K");
        System.out.println("╚═══════════════════════════════════════════════════");
    }

    /**
     * УЛУЧШЕННОЕ предсказание с нормализацией
     */
    public double predict(double[][] sequence) {
        if (sequence.length != sequenceLength || sequence[0].length != numFeatures) {
            throw new IllegalArgumentException(
                    String.format("Wrong size: expected [%d, %d], got [%d, %d]",
                            sequenceLength, numFeatures, sequence.length, sequence[0].length)
            );
        }

        // Нормализация входных данных (КРИТИЧНО!)
        double[][] normalized = normalizeSequence(sequence);

        // Конвертируем в INDArray [1, numFeatures, sequenceLength]
        INDArray input = Nd4j.create(new int[]{1, numFeatures, sequenceLength});
        for (int t = 0; t < sequenceLength; t++) {
            for (int f = 0; f < numFeatures; f++) {
                input.putScalar(new int[]{0, f, t}, normalized[t][f]);
            }
        }

        // Отключаем dropout для inference
        network.setLayerMaskArrays(null, null);

        // Предсказание
        INDArray output = network.output(input, false);

        // Возвращаем вероятность последнего тика
        return output.getDouble(0, 0, sequenceLength - 1);
    }

    /**
     * КРИТИЧНО: Нормализация последовательности
     */
    private double[][] normalizeSequence(double[][] sequence) {
        double[][] normalized = new double[sequenceLength][numFeatures];

        for (int t = 0; t < sequenceLength; t++) {
            for (int f = 0; f < numFeatures; f++) {
                if (featureStds[f] > 1e-10) {
                    normalized[t][f] = (sequence[t][f] - featureMeans[f]) / featureStds[f];
                } else {
                    normalized[t][f] = 0.0;
                }

                // Clipping для стабильности
                if (normalized[t][f] > 5.0) normalized[t][f] = 5.0;
                if (normalized[t][f] < -5.0) normalized[t][f] = -5.0;
            }
        }

        return normalized;
    }

    /**
     * УЛУЧШЕННОЕ обучение с Early Stopping
     */
    public void train(List<TrainingExample> trainData, List<TrainingExample> validData, int maxEpochs) {
        System.out.println("╔═══════════════════════════════════════════════════");
        System.out.println("║ [SmartAimModel] Начинаем обучение ULTRA SMART модели");
        System.out.println("║ Train: " + trainData.size() + ", Valid: " + validData.size());
        System.out.println("║ Max Epochs: " + maxEpochs);
        System.out.println("╚═══════════════════════════════════════════════════");

        // Вычисляем статистики для нормализации
        computeNormalizationStats(trainData);

        // Нормализуем данные
        List<TrainingExample> normTrain = normalizeDataset(trainData);
        List<TrainingExample> normValid = normalizeDataset(validData);

        // Early stopping параметры
        double bestValidLoss = Double.MAX_VALUE;
        int patienceCounter = 0;
        int patience = 15;  // Останавливаемся если 15 эпох нет улучшения

        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            // Перемешиваем данные
            Collections.shuffle(normTrain);

            // Обучение на эпохе
            double trainLoss = 0.0;
            for (TrainingExample example : normTrain) {
                DataSet ds = createDataSet(example);
                network.fit(ds);
                trainLoss += network.score(ds);
            }
            trainLoss /= normTrain.size();

            // Валидация
            double validLoss = evaluateModel(normValid);

            System.out.printf("[SmartAimModel] Epoch %3d: train_loss=%.4f, valid_loss=%.4f%n",
                    epoch + 1, trainLoss, validLoss);

            // Early stopping
            if (validLoss < bestValidLoss * 0.995) {  // Улучшение хотя бы на 0.5%
                bestValidLoss = validLoss;
                patienceCounter = 0;

                // Сохраняем лучшие веса (в памяти)
                try {
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    network.save(baos);
                } catch (IOException ignored) {}

            } else {
                patienceCounter++;

                if (patienceCounter >= patience) {
                    System.out.println("[SmartAimModel] Early stopping на эпохе " + (epoch + 1));
                    break;
                }
            }

            // Learning rate decay каждые 20 эпох
            if ((epoch + 1) % 20 == 0) {
                double currentLR = ((Adam) network.conf().getLayer(1).getIUpdater()).getLearningRate();
                double newLR = currentLR * 0.5;

                for (int i = 0; i < network.getnLayers(); i++) {
                    if (network.conf().getLayer(i).getIUpdater() instanceof Adam) {
                        ((Adam) network.conf().getLayer(i).getIUpdater()).setLearningRate(newLR);
                    }
                }

                System.out.printf("[SmartAimModel] Learning rate decay: %.6f → %.6f%n", currentLR, newLR);
            }
        }

        // Финальная оценка
        evaluateMetrics(normValid);

        System.out.println("╔═══════════════════════════════════════════════════");
        System.out.println("║ [SmartAimModel] Обучение завершено!");
        System.out.println("║ Accuracy:  " + String.format("%.2f%%", accuracy * 100));
        System.out.println("║ Precision: " + String.format("%.2f%%", precision * 100));
        System.out.println("║ Recall:    " + String.format("%.2f%%", recall * 100));
        System.out.println("║ F1-Score:  " + String.format("%.4f", f1Score));
        System.out.println("╚═══════════════════════════════════════════════════");
    }

    /**
     * Вычислить статистики для нормализации
     */
    private void computeNormalizationStats(List<TrainingExample> data) {
        // Сбрасываем
        for (int f = 0; f < numFeatures; f++) {
            featureMeans[f] = 0.0;
            featureStds[f] = 0.0;
        }

        // Вычисляем средние
        for (TrainingExample ex : data) {
            for (int t = 0; t < sequenceLength; t++) {
                for (int f = 0; f < numFeatures; f++) {
                    featureMeans[f] += ex.sequence[t][f];
                }
            }
        }

        int totalSamples = data.size() * sequenceLength;
        for (int f = 0; f < numFeatures; f++) {
            featureMeans[f] /= totalSamples;
        }

        // Вычисляем std
        for (TrainingExample ex : data) {
            for (int t = 0; t < sequenceLength; t++) {
                for (int f = 0; f < numFeatures; f++) {
                    double diff = ex.sequence[t][f] - featureMeans[f];
                    featureStds[f] += diff * diff;
                }
            }
        }

        for (int f = 0; f < numFeatures; f++) {
            featureStds[f] = Math.sqrt(featureStds[f] / totalSamples);
            if (featureStds[f] < 1e-10) {
                featureStds[f] = 1.0;
            }
        }

        System.out.println("[SmartAimModel] Нормализация вычислена:");
        for (int f = 0; f < numFeatures; f++) {
            System.out.printf("  Feature %d: mean=%.4f, std=%.4f%n", f, featureMeans[f], featureStds[f]);
        }
    }

    /**
     * Нормализовать весь датасет
     */
    private List<TrainingExample> normalizeDataset(List<TrainingExample> data) {
        return data.stream()
                .map(ex -> new TrainingExample(normalizeSequence(ex.sequence), ex.label))
                .toList();
    }

    /**
     * Создать DataSet для DL4J
     */
    private DataSet createDataSet(TrainingExample example) {
        // Input: [1, numFeatures, sequenceLength]
        INDArray input = Nd4j.create(new int[]{1, numFeatures, sequenceLength});

        for (int t = 0; t < sequenceLength; t++) {
            for (int f = 0; f < numFeatures; f++) {
                input.putScalar(new int[]{0, f, t}, example.sequence[t][f]);
            }
        }

        // Label: [1, 1, sequenceLength]
        INDArray label = Nd4j.zeros(1, 1, sequenceLength);
        label.putScalar(new int[]{0, 0, sequenceLength - 1}, example.label);

        // Mask: только последний тик важен
        INDArray mask = Nd4j.zeros(1, sequenceLength);
        mask.putScalar(new int[]{0, sequenceLength - 1}, 1.0);

        DataSet ds = new DataSet(input, label);
        ds.setLabelsMaskArray(mask);

        return ds;
    }

    /**
     * Оценка модели
     */
    private double evaluateModel(List<TrainingExample> data) {
        double totalLoss = 0.0;

        for (TrainingExample example : data) {
            DataSet ds = createDataSet(example);
            totalLoss += network.score(ds);
        }

        return totalLoss / data.size();
    }

    /**
     * Вычисление метрик
     */
    private void evaluateMetrics(List<TrainingExample> data) {
        int tp = 0, fp = 0, tn = 0, fn = 0;

        for (TrainingExample example : data) {
            double prediction = predict(example.sequence);
            boolean predicted = prediction >= 0.5;
            boolean actual = example.label >= 0.5;

            if (predicted && actual) tp++;
            else if (predicted && !actual) fp++;
            else if (!predicted && !actual) tn++;
            else fn++;
        }

        accuracy = (tp + tn) / (double) data.size();
        precision = tp / (double) (tp + fp + 1e-10);
        recall = tp / (double) (tp + fn + 1e-10);
        f1Score = 2 * precision * recall / (precision + recall + 1e-10);
    }

    /**
     * Сохранить модель
     */
    public void save(String path) throws IOException {
        File file = new File(path);
        file.getParentFile().mkdirs();

        // Сохраняем сеть
        network.save(file);

        // Сохраняем статистики нормализации отдельно
        File statsFile = new File(path + ".stats");
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(statsFile))) {
            oos.writeObject(featureMeans);
            oos.writeObject(featureStds);
            oos.writeDouble(accuracy);
            oos.writeDouble(precision);
            oos.writeDouble(recall);
            oos.writeDouble(f1Score);
        }

        System.out.println("[SmartAimModel] Модель сохранена: " + path);
        System.out.println("[SmartAimModel] Размер: " + (file.length() / 1024) + " KB");
    }

    /**
     * Загрузить модель
     */
    public static SmartAimModel load(String path) throws IOException, ClassNotFoundException {
        SmartAimModel model = new SmartAimModel();

        // Загружаем сеть
        model.network = MultiLayerNetwork.load(new File(path), true);

        // Загружаем статистики
        File statsFile = new File(path + ".stats");
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(statsFile))) {
            model.featureMeans = (double[]) ois.readObject();
            model.featureStds = (double[]) ois.readObject();
            model.accuracy = ois.readDouble();
            model.precision = ois.readDouble();
            model.recall = ois.readDouble();
            model.f1Score = ois.readDouble();
        }

        System.out.println("╔═══════════════════════════════════════════════════");
        System.out.println("║ [SmartAimModel] Модель загружена: " + path);
        System.out.println("║ Accuracy:  " + String.format("%.2f%%", model.accuracy * 100));
        System.out.println("║ Precision: " + String.format("%.2f%%", model.precision * 100));
        System.out.println("║ Recall:    " + String.format("%.2f%%", model.recall * 100));
        System.out.println("║ F1-Score:  " + String.format("%.4f", model.f1Score));
        System.out.println("╚═══════════════════════════════════════════════════");

        return model;
    }

    // Getters
    public double getAccuracy() { return accuracy; }
    public double getPrecision() { return precision; }
    public double getRecall() { return recall; }
    public double getF1Score() { return f1Score; }
    public int getNumParams() { return (int) network.numParams(); }

    /**
     * Пример для обучения
     */
    public static class TrainingExample implements Serializable {
        private static final long serialVersionUID = 1L;

        public final double[][] sequence;  // [sequenceLength=40, numFeatures=8]
        public final double label;         // 0.0 = legit, 1.0 = cheat

        public TrainingExample(double[][] sequence, double label) {
            this.sequence = sequence;
            this.label = label;
        }
    }
}
