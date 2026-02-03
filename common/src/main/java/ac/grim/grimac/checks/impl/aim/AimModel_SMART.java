package ac.grim.grimac.checks.impl.aim;

import java.io.*;
import java.util.*;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * УЛУЧШЕННАЯ ML-модель для детекции аимботов
 *
 * ИСПРАВЛЕНИЯ:
 * 1. ✅ Thread-safe операции
 * 2. ✅ Proper exception handling вместо silent fail
 * 3. ✅ Model versioning для совместимости
 * 4. ✅ Feature validation
 * 5. ✅ Метрики производительности
 *
 * @author ImprovedAImML Team
 * @version 3.0 (улучшено на базе MLSAC)
 */
public class AimModel_SMART implements Serializable {
    private static final long serialVersionUID = 3L;

    // Model version для проверки совместимости
    private static final int MODEL_VERSION = 3;
    private final int version;

    // Параметры модели
    private double[] weights;
    private double bias;
    private double[] featureMeans;
    private double[] featureStds;

    // Метрики обучения
    private double accuracy;
    private double precision;
    private double recall;
    private double f1Score;
    private int trainingIterations;
    private double finalLoss;

    // Thread safety
    private transient ReadWriteLock lock;

    // Performance metrics
    private transient long totalPredictions = 0;
    private transient long totalPredictionTime = 0;

    /**
     * Custom exception для проблем с размерностью
     */
    public static class ModelDimensionMismatchException extends RuntimeException {
        public ModelDimensionMismatchException(String message) {
            super(message);
        }
    }

    /**
     * Custom exception для проблем с версией
     */
    public static class ModelVersionMismatchException extends RuntimeException {
        public ModelVersionMismatchException(String message) {
            super(message);
        }
    }

    public AimModel_SMART(int numFeatures) {
        this.version = MODEL_VERSION;
        this.weights = new double[numFeatures];
        this.bias = 0.0;
        this.featureMeans = new double[numFeatures];
        this.featureStds = new double[numFeatures];
        this.lock = new ReentrantReadWriteLock();

        // Xavier initialization для лучшей сходимости
        Random rand = new Random(42);
        double scale = Math.sqrt(2.0 / numFeatures);
        for (int i = 0; i < numFeatures; i++) {
            weights[i] = (rand.nextGaussian()) * scale;
        }
    }

    /**
     * Сигмоида с защитой от overflow
     */
    private double sigmoid(double z) {
        // Защита от overflow
        if (z > 20) return 1.0;
        if (z < -20) return 0.0;
        return 1.0 / (1.0 + Math.exp(-z));
    }

    /**
     * Предсказание для нормализованных фичей
     */
    private double predictRaw(double[] features) {
        lock.readLock().lock();
        try {
            double z = bias;
            for (int i = 0; i < Math.min(features.length, weights.length); i++) {
                z += weights[i] * features[i];
            }
            return sigmoid(z);
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * УЛУЧШЕННОЕ публичное предсказание с validation и metrics
     */
    public double predict(double[] features) {
        long startTime = System.nanoTime();

        try {
            // Validation
            if (features == null || features.length == 0) {
                throw new IllegalArgumentException("Features cannot be null or empty");
            }

            // КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Бросаем exception вместо silent fail
            if (features.length != weights.length) {
                throw new ModelDimensionMismatchException(
                        String.format(
                                "Model dimension mismatch! Model expects %d features, but got %d. " +
                                        "Please retrain the model with /grimaitrain",
                                weights.length, features.length
                        )
                );
            }

            double[] normalized = normalizeFeatures(features);
            double result = predictRaw(normalized);

            // Update metrics
            totalPredictions++;
            totalPredictionTime += (System.nanoTime() - startTime);

            return result;

        } catch (ModelDimensionMismatchException e) {
            // Log detailed error
            System.err.println("╔════════════════════════════════════════════╗");
            System.err.println("║  MODEL DIMENSION MISMATCH ERROR!          ║");
            System.err.println("╠════════════════════════════════════════════╣");
            System.err.println("║  Expected features: " + weights.length);
            System.err.println("║  Received features: " + features.length);
            System.err.println("║  ⚠ Model is NOT working!                  ║");
            System.err.println("║  ✓ Solution: /grimaitrain                 ║");
            System.err.println("╚════════════════════════════════════════════╝");
            throw e; // Re-throw для proper handling
        }
    }

    /**
     * УЛУЧШЕННАЯ нормализация с validation
     */
    private double[] normalizeFeatures(double[] features) {
        if (features.length != featureMeans.length) {
            throw new ModelDimensionMismatchException(
                    String.format(
                            "Cannot normalize: Model trained on %d features, got %d",
                            featureMeans.length, features.length
                    )
            );
        }

        double[] normalized = new double[features.length];

        for (int i = 0; i < features.length; i++) {
            // Проверка на NaN/Infinity
            if (Double.isNaN(features[i]) || Double.isInfinite(features[i])) {
                System.err.println("Warning: Invalid feature value at index " + i + ": " + features[i]);
                normalized[i] = 0.0;
                continue;
            }

            if (featureStds[i] > 1e-10) {
                normalized[i] = (features[i] - featureMeans[i]) / featureStds[i];
            } else {
                normalized[i] = 0.0;
            }

            // Clipping для stability
            if (normalized[i] > 10.0) normalized[i] = 10.0;
            if (normalized[i] < -10.0) normalized[i] = -10.0;
        }

        return normalized;
    }

    /**
     * УЛУЧШЕННОЕ обучение с early stopping и adaptive learning rate
     */
    public void train(List<TrainingExample> trainingData, List<TrainingExample> validationData,
                      double initialLearningRate, double l2Lambda, int maxIterations) {

        lock.writeLock().lock();
        try {
            if (trainingData.isEmpty()) {
                throw new IllegalArgumentException("Training data is empty!");
            }

            System.out.println("[GrimAC ML] ═══════════════════════════════════");
            System.out.println("[GrimAC ML] Starting model training...");
            System.out.println("[GrimAC ML] Training samples: " + trainingData.size());
            System.out.println("[GrimAC ML] Validation samples: " + validationData.size());
            System.out.println("[GrimAC ML] ═══════════════════════════════════");

            // Compute normalization statistics
            computeNormalizationStats(trainingData);

            // Normalize data
            List<TrainingExample> normalizedTrain = normalizeDataset(trainingData);
            List<TrainingExample> normalizedValid = normalizeDataset(validationData);

            // Training loop with adaptive learning rate
            double learningRate = initialLearningRate;
            double bestValidLoss = Double.MAX_VALUE;
            double[] bestWeights = null;
            double bestBias = 0.0;
            int patience = 50;
            int patienceCounter = 0;
            int bestIteration = 0;

            for (int iter = 0; iter < maxIterations; iter++) {
                // Shuffle training data
                Collections.shuffle(normalizedTrain);

                // One epoch
                double totalLoss = 0.0;

                for (TrainingExample example : normalizedTrain) {
                    // Forward pass
                    double prediction = predictRaw(example.features);
                    double error = prediction - example.label;

                    // Backward pass with L2 regularization
                    for (int i = 0; i < weights.length; i++) {
                        double gradient = error * example.features[i] + l2Lambda * weights[i];
                        weights[i] -= learningRate * gradient;
                    }
                    bias -= learningRate * error;

                    // Compute loss (binary cross-entropy)
                    double loss = -example.label * Math.log(Math.max(prediction, 1e-10))
                            - (1 - example.label) * Math.log(Math.max(1 - prediction, 1e-10));
                    totalLoss += loss;
                }

                // Validation every 10 iterations
                if (iter % 10 == 0) {
                    double validLoss = computeLoss(normalizedValid);
                    double trainLoss = totalLoss / normalizedTrain.size();

                    System.out.printf("[GrimAC ML] Iter %d: train_loss=%.4f, valid_loss=%.4f, lr=%.6f%n",
                            iter, trainLoss, validLoss, learningRate);

                    // Early stopping
                    if (validLoss < bestValidLoss) {
                        bestValidLoss = validLoss;
                        bestWeights = weights.clone();
                        bestBias = bias;
                        bestIteration = iter;
                        patienceCounter = 0;
                    } else {
                        patienceCounter++;

                        // Adaptive learning rate
                        if (patienceCounter % 10 == 0) {
                            learningRate *= 0.5;
                            System.out.println("[GrimAC ML] Reducing learning rate to: " + learningRate);
                        }

                        if (patienceCounter >= patience) {
                            System.out.println("[GrimAC ML] Early stopping at iteration " + iter);
                            break;
                        }
                    }
                }
            }

            // Restore best weights
            if (bestWeights != null) {
                weights = bestWeights;
                bias = bestBias;
                System.out.println("[GrimAC ML] Restored best model from iteration " + bestIteration);
            }

            // Final evaluation
            evaluateModel(normalizedValid);

            this.trainingIterations = maxIterations;
            this.finalLoss = bestValidLoss;

            // Print final metrics
            System.out.println("[GrimAC ML] ═══════════════════════════════════");
            System.out.println("[GrimAC ML] Training complete!");
            System.out.println("[GrimAC ML] Best iteration: " + bestIteration);
            System.out.println("[GrimAC ML] Accuracy: " + String.format("%.2f%%", accuracy * 100));
            System.out.println("[GrimAC ML] Precision: " + String.format("%.2f%%", precision * 100));
            System.out.println("[GrimAC ML] Recall: " + String.format("%.2f%%", recall * 100));
            System.out.println("[GrimAC ML] F1-Score: " + String.format("%.4f", f1Score));
            System.out.println("[GrimAC ML] Final Loss: " + String.format("%.4f", finalLoss));
            System.out.println("[GrimAC ML] ═══════════════════════════════════");

        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Normalize entire dataset
     */
    private List<TrainingExample> normalizeDataset(List<TrainingExample> data) {
        List<TrainingExample> normalized = new ArrayList<>(data.size());
        for (TrainingExample ex : data) {
            normalized.add(new TrainingExample(
                    normalizeFeatures(ex.features),
                    ex.label
            ));
        }
        return normalized;
    }

    /**
     * Compute normalization statistics
     */
    private void computeNormalizationStats(List<TrainingExample> data) {
        int numFeatures = data.get(0).features.length;

        // Compute means
        for (TrainingExample ex : data) {
            for (int i = 0; i < numFeatures; i++) {
                featureMeans[i] += ex.features[i];
            }
        }
        for (int i = 0; i < numFeatures; i++) {
            featureMeans[i] /= data.size();
        }

        // Compute standard deviations
        for (TrainingExample ex : data) {
            for (int i = 0; i < numFeatures; i++) {
                double diff = ex.features[i] - featureMeans[i];
                featureStds[i] += diff * diff;
            }
        }
        for (int i = 0; i < numFeatures; i++) {
            featureStds[i] = Math.sqrt(featureStds[i] / data.size());
            // Prevent division by zero
            if (featureStds[i] < 1e-10) {
                featureStds[i] = 1.0;
            }
        }
    }

    /**
     * Compute loss on dataset
     */
    private double computeLoss(List<TrainingExample> data) {
        double totalLoss = 0.0;
        for (TrainingExample ex : data) {
            double pred = predictRaw(ex.features);
            double loss = -ex.label * Math.log(Math.max(pred, 1e-10))
                    - (1 - ex.label) * Math.log(Math.max(1 - pred, 1e-10));
            totalLoss += loss;
        }
        return totalLoss / data.size();
    }

    /**
     * Evaluate model on validation set
     */
    private void evaluateModel(List<TrainingExample> validData) {
        int truePositives = 0;
        int falsePositives = 0;
        int trueNegatives = 0;
        int falseNegatives = 0;

        for (TrainingExample ex : validData) {
            double pred = predictRaw(ex.features);
            boolean predicted = pred >= 0.5;
            boolean actual = ex.label >= 0.5;

            if (predicted && actual) truePositives++;
            else if (predicted && !actual) falsePositives++;
            else if (!predicted && !actual) trueNegatives++;
            else falseNegatives++;
        }

        this.accuracy = (truePositives + trueNegatives) / (double) validData.size();

        if (truePositives + falsePositives > 0) {
            this.precision = truePositives / (double) (truePositives + falsePositives);
        } else {
            this.precision = 0.0;
        }

        if (truePositives + falseNegatives > 0) {
            this.recall = truePositives / (double) (truePositives + falseNegatives);
        } else {
            this.recall = 0.0;
        }

        if (precision + recall > 0) {
            this.f1Score = 2 * (precision * recall) / (precision + recall);
        } else {
            this.f1Score = 0.0;
        }
    }

    /**
     * УЛУЧШЕННОЕ сохранение с validation
     */
    public void save(String path) throws IOException {
        lock.readLock().lock();
        try {
            File file = new File(path);
            file.getParentFile().mkdirs();

            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
                oos.writeObject(this);
            }

            System.out.println("[GrimAC ML] Model saved: " + path);
            System.out.println("[GrimAC ML] Model version: " + version);
            System.out.println("[GrimAC ML] Feature count: " + weights.length);

        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * УЛУЧШЕННАЯ загрузка с version check
     */
    public static AimModel_SMART load(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            AimModel_SMART model = (AimModel_SMART) ois.readObject();

            // Version compatibility check
            if (model.version != MODEL_VERSION) {
                System.err.println("╔════════════════════════════════════════════╗");
                System.err.println("║  MODEL VERSION MISMATCH WARNING!          ║");
                System.err.println("╠════════════════════════════════════════════╣");
                System.err.println("║  Model version: " + model.version);
                System.err.println("║  Current version: " + MODEL_VERSION);
                System.err.println("║  ⚠ Consider retraining the model          ║");
                System.err.println("╚════════════════════════════════════════════╝");
            }

            // Re-initialize transient fields
            model.lock = new ReentrantReadWriteLock();
            model.totalPredictions = 0;
            model.totalPredictionTime = 0;

            System.out.println("[GrimAC ML] Model loaded: " + path);
            System.out.println("[GrimAC ML] Feature count: " + model.weights.length);
            System.out.println("[GrimAC ML] " + model.getMetrics());

            return model;
        }
    }

    /**
     * Get model metrics
     */
    public String getMetrics() {
        return String.format(
                "Accuracy: %.2f%% | Precision: %.2f%% | Recall: %.2f%% | F1: %.4f",
                accuracy * 100, precision * 100, recall * 100, f1Score
        );
    }

    /**
     * Get performance metrics
     */
    public String getPerformanceMetrics() {
        if (totalPredictions == 0) {
            return "No predictions yet";
        }

        double avgTimeMs = (totalPredictionTime / (double) totalPredictions) / 1_000_000.0;
        return String.format(
                "Total predictions: %d | Avg time: %.3f ms",
                totalPredictions, avgTimeMs
        );
    }

    // Getters
    public double getAccuracy() { return accuracy; }
    public double getPrecision() { return precision; }
    public double getRecall() { return recall; }
    public double getF1Score() { return f1Score; }
    public int getNumFeatures() { return weights.length; }
    public int getVersion() { return version; }

    /**
     * Training example class
     */
    public static class TrainingExample implements Serializable {
        private static final long serialVersionUID = 1L;

        public final double[] features;
        public final double label; // 0.0 = legit, 1.0 = cheat

        public TrainingExample(double[] features, double label) {
            this.features = features;
            this.label = label;
        }
    }
}
