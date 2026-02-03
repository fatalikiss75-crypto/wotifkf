package ac.grim.grimac.checks.impl.aim;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.*;

/**
 * Модель ML для предсказания человеческой моторики движения камеры (Sequence-to-One Regression)
 *
 * Главная задача:
 * По последовательности из 40 тиков движения камеры (yaw/pitch и производные)
 * предсказать, какими будут deltaYaw/deltaPitch на следующем тике.
 *
 * Архитектура:
 * - BatchNormalization (нормализация входа)
 * - LSTM 256 нейронов + Dropout 0.3
 * - LSTM 128 нейронов + Dropout 0.3
 * - Dense 64 нейрона + ReLU + Dropout 0.4
 * - Dense 32 нейрона + ReLU + Dropout 0.4
 * - Output 2 нейрона (TANH) - предсказывает [nextDeltaYaw, nextDeltaPitch]
 *
 * Loss: MSE (Mean Squared Error)
 *
 * Экспорт:
 * Все веса экспортируются в JSON для загрузки в Grim без DL4J зависимости.
 *
 * @author GrimAC ML Team
 * @version 2.0 - Regression Edition
 */
public class SmartAimModel implements Serializable {
    private static final long serialVersionUID = 2L;

    private transient MultiLayerNetwork network;

    // Параметры
    private final int sequenceLength = 40;
    private final int numFeatures = 8;  // deltaYaw, deltaPitch, accel*2, jerk*2, gcdError*2

    // Метрики регрессии
    private double validationMSE = 0.0;
    private double validationMAE = 0.0;

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
     * Построить нейронную сеть для регрессии
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

                // Выходной слой: 2 нейрона (предсказание [nextDeltaYaw, nextDeltaPitch])
                .layer(9, new RnnOutputLayer.Builder()
                        .nIn(32)
                        .nOut(2)  // 2 выхода: deltaYaw, deltaPitch
                        .activation(Activation.TANH)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())

                .build();

        network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(new ScoreIterationListener(5));

        System.out.println("╔═══════════════════════════════════════════════════");
        System.out.println("║ [SmartAimModel] Сеть регрессии создана!");
        System.out.println("║ BatchNorm → LSTM(256) → Dropout → LSTM(128)");
        System.out.println("║ → Dropout → Dense(64) → Dense(32) → Output(2)");
        System.out.println("║ Activation: TANH | Loss: MSE");
        System.out.println("║ Параметров: ~" + (network.numParams() / 1000) + "K");
        System.out.println("╚═══════════════════════════════════════════════════");
    }

    /**
     * Предсказание следующего deltaYaw/deltaPitch
     *
     * @param sequence [40][8] - последовательность из 40 тиков
     * @return double[2] - [predictedDeltaYaw, predictedDeltaPitch]
     */
    public double[] predict(double[][] sequence) {
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

        // Возвращаем предсказание [nextDeltaYaw, nextDeltaPitch]
        double[] result = new double[2];
        result[0] = output.getDouble(0, 0, sequenceLength - 1);  // deltaYaw
        result[1] = output.getDouble(0, 1, sequenceLength - 1);  // deltaPitch

        return result;
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

                // Clipping для стабильности [-5, 5]
                if (normalized[t][f] > 5.0) normalized[t][f] = 5.0;
                if (normalized[t][f] < -5.0) normalized[t][f] = -5.0;
            }
        }

        return normalized;
    }

    /**
     * Обучение с Early Stopping и Learning Rate Decay
     */
    public void train(List<TrainingExample> trainData, List<TrainingExample> validData, int maxEpochs) {
        System.out.println("╔═══════════════════════════════════════════════════");
        System.out.println("║ [SmartAimModel] Начинаем обучение регрессии");
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

        // Сортируем данные в батчи для обучения
        int batchSize = 32;
        ListDataSetIterator<DataSet> trainIter = createDataIterator(normTrain, batchSize, true);
        ListDataSetIterator<DataSet> validIter = createDataIterator(normValid, batchSize, false);

        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            // Перемешиваем и обучаем на эпохе
            trainIter.reset();
            trainIter.shuffle();
            while (trainIter.hasNext()) {
                DataSet ds = trainIter.next();
                network.fit(ds);
            }

            // Вычисляем loss на train и valid
            double trainLoss = computeAverageLoss(normTrain);
            double validLoss = computeAverageLoss(normValid);

            // Вычисляем MAE на valid
            double validMAE = computeAverageMAE(normValid);
            this.validationMAE = validMAE;

            System.out.printf("[SmartAimModel] Epoch %3d: train_MSE=%.6f, valid_MSE=%.6f, valid_MAE=%.6f%n",
                    epoch + 1, trainLoss, validLoss, validMAE);

            // Early stopping
            if (validLoss < bestValidLoss * 0.995) {  // Улучшение хотя бы на 0.5%
                bestValidLoss = validLoss;
                this.validationMSE = validLoss;
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
        finalEvaluation(normValid);

        System.out.println("╔═══════════════════════════════════════════════════");
        System.out.println("║ [SmartAimModel] Обучение завершено!");
        System.out.println("║ Valid MSE:  " + String.format("%.6f", validationMSE));
        System.out.println("║ Valid MAE:  " + String.format("%.6f", validationMAE));
        System.out.println("╚═══════════════════════════════════════════════════");
    }

    /**
     * Создать итератор данных для обучения
     */
    private ListDataSetIterator<DataSet> createDataIterator(List<TrainingExample> data, int batchSize, boolean shuffle) {
        List<DataSet> datasets = new ArrayList<>();
        for (TrainingExample example : data) {
            datasets.add(createDataSet(example));
        }
        if (shuffle) {
            Collections.shuffle(datasets);
        }
        return new ListDataSetIterator<>(datasets, batchSize);
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

        // Label: [1, 2, 1] - предсказываем 2 значения в последний момент времени
        INDArray label = Nd4j.create(new int[]{1, 2, 1});
        label.putScalar(new int[]{0, 0, 0}, example.label[0]);  // nextDeltaYaw
        label.putScalar(new int[]{0, 1, 0}, example.label[1]);  // nextDeltaPitch

        return new DataSet(input, label);
    }

    /**
     * Вычислить средний loss (MSE) на датасете
     */
    private double computeAverageLoss(List<TrainingExample> data) {
        double totalLoss = 0.0;

        for (TrainingExample example : data) {
            DataSet ds = createDataSet(example);
            totalLoss += network.score(ds);
        }

        return totalLoss / data.size();
    }

    /**
     * Вычислить средний MAE (Mean Absolute Error) на датасете
     */
    private double computeAverageMAE(List<TrainingExample> data) {
        double totalError = 0.0;

        for (TrainingExample example : data) {
            double[] prediction = predict(example.sequence);

            // Денормализуем предсказание для честной оценки
            double denormYaw = denormalizeFeature(prediction[0], 0);
            double denormPitch = denormalizeFeature(prediction[1], 1);

            double denormTargetYaw = denormalizeFeature(example.label[0], 0);
            double denormTargetPitch = denormalizeFeature(example.label[1], 1);

            double errorYaw = Math.abs(denormYaw - denormTargetYaw);
            double errorPitch = Math.abs(denormPitch - denormTargetPitch);

            totalError += errorYaw + errorPitch;
        }

        return totalError / (2 * data.size());  // Среднее по 2 выходам
    }

    /**
     * Денормализация отдельного фичи
     */
    private double denormalizeFeature(double normalized, int featureIndex) {
        return normalized * featureStds[featureIndex] + featureMeans[featureIndex];
    }

    /**
     * Финальная оценка модели
     */
    private void finalEvaluation(List<TrainingExample> data) {
        this.validationMSE = computeAverageLoss(data);
        this.validationMAE = computeAverageMAE(data);
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
     * Экспортировать веса и нормализацию в JSON для использования в Grim
     */
    public void exportToJson(String path) throws IOException {
        File file = new File(path);
        file.getParentFile().mkdirs();

        // Создаём объект для экспорта
        ModelExport export = new ModelExport();

        // LSTM слой 1 (256 нейронов)
        export.lstm1_W = extractWeights(network.getLayer(1).getParam("W")).toDoubleVector();
        export.lstm1_RW = extractWeights(network.getLayer(1).getParam("RW")).toDoubleVector();
        export.lstm1_b = extractWeights(network.getLayer(1).getParam("b")).toDoubleVector();

        // LSTM слой 2 (128 нейронов)
        export.lstm2_W = extractWeights(network.getLayer(3).getParam("W")).toDoubleVector();
        export.lstm2_RW = extractWeights(network.getLayer(3).getParam("RW")).toDoubleVector();
        export.lstm2_b = extractWeights(network.getLayer(3).getParam("b")).toDoubleVector();

        // Dense слой 1 (64 нейрона)
        export.dense1_W = extractWeights(network.getLayer(5).getParam("W")).toDoubleVector();
        export.dense1_b = extractWeights(network.getLayer(5).getParam("b")).toDoubleVector();

        // Dense слой 2 (32 нейрона)
        export.dense2_W = extractWeights(network.getLayer(7).getParam("W")).toDoubleVector();
        export.dense2_b = extractWeights(network.getLayer(7).getParam("b")).toDoubleVector();

        // Output слой (2 нейрона)
        export.out_W = extractWeights(network.getLayer(9).getParam("W")).toDoubleVector();
        export.out_b = extractWeights(network.getLayer(9).getParam("b")).toDoubleVector();

        // Нормализация
        export.featureMeans = featureMeans;
        export.featureStds = featureStds;

        // Метаданные
        export.version = "2.0";
        export.validationMSE = validationMSE;
        export.validationMAE = validationMAE;

        // Сохраняем в JSON
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        try (FileWriter writer = new FileWriter(file)) {
            gson.toJson(export, writer);
        }

        System.out.println("[SmartAimModel] Веса экспортированы: " + path);
        System.out.println("[SmartAimModel] Размер файла: " + (file.length() / 1024) + " KB");
    }

    /**
     * Извлечь веса из INDArray
     */
    private INDArray extractWeights(INDArray param) {
        return param.dup();
    }

    /**
     * Сохранить модель (старый метод, сохраняет model.zip)
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
            oos.writeDouble(validationMSE);
            oos.writeDouble(validationMAE);
        }

        System.out.println("[SmartAimModel] Модель сохранена: " + path);
        System.out.println("[SmartAimModel] Размер: " + (file.length() / 1024) + " KB");
    }

    /**
     * Загрузить модель (старый метод)
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
            model.validationMSE = ois.readDouble();
            model.validationMAE = ois.readDouble();
        }

        System.out.println("╔═══════════════════════════════════════════════════");
        System.out.println("║ [SmartAimModel] Модель загружена: " + path);
        System.out.println("║ Valid MSE:  " + String.format("%.6f", model.validationMSE));
        System.out.println("║ Valid MAE:  " + String.format("%.6f", model.validationMAE));
        System.out.println("╚═══════════════════════════════════════════════════");

        return model;
    }

    // Getters
    public double getValidationMSE() { return validationMSE; }
    public double getValidationMAE() { return validationMAE; }
    public double[] getFeatureMeans() { return featureMeans; }
    public double[] getFeatureStds() { return featureStds; }
    public int getNumParams() { return (int) network.numParams(); }

    /**
     * Класс для экспорта модели в JSON
     */
    public static class ModelExport {
        public String version;
        public double[] lstm1_W;
        public double[] lstm1_RW;
        public double[] lstm1_b;
        public double[] lstm2_W;
        public double[] lstm2_RW;
        public double[] lstm2_b;
        public double[] dense1_W;
        public double[] dense1_b;
        public double[] dense2_W;
        public double[] dense2_b;
        public double[] out_W;
        public double[] out_b;
        public double[] featureMeans;
        public double[] featureStds;
        public double validationMSE;
        public double validationMAE;
    }

    /**
     * Пример для обучения
     */
    public static class TrainingExample implements Serializable {
        private static final long serialVersionUID = 2L;

        public final double[][] sequence;  // [sequenceLength=40, numFeatures=8]
        public final double[] label;      // [nextDeltaYaw, nextDeltaPitch]

        public TrainingExample(double[][] sequence, double[] label) {
            this.sequence = sequence;
            this.label = label;
        }
    }
}
