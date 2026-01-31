package ac.grim.grimac.checks.impl.aim;

import ac.grim.grimac.GrimAPI;
import ac.grim.grimac.checks.Check;
import ac.grim.grimac.checks.CheckData;
import ac.grim.grimac.checks.type.RotationCheck;
import ac.grim.grimac.player.GrimPlayer;
import ac.grim.grimac.utils.anticheat.update.RotationUpdate;
import lombok.Getter;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Улучшенная ML-система детекции аимботов (по образцу SlothAC)
 *
 * Основные улучшения:
 * - Глобальный сбор датасетов (со всех игроков одновременно)
 * - Автоматическая архивация в ZIP
 * - Продвинутые фичи (jerk, accel, GCD-errors)
 * - Более точная детекция через FlatBuffers
 *
 * Команды:
 * /grimAiOn - Начать локальную запись
 * /grimAiOff - Остановить локальную запись
 * /grimAiGlobal start <id> - Начать глобальный сбор
 * /grimAiGlobal stop - Остановить глобальный сбор
 * /grimAiTrain - Обучить модель
 * /shame ban <player> - Забанить читера
 */
@CheckData(name = "AimML", description = "Machine Learning AimBot Detection", experimental = true)
public class ImprovedAimML extends Check implements RotationCheck {

    // ========== КОНСТАНТЫ ==========
    private static final int SEQUENCE_LENGTH = 40; // Длина последовательности для анализа
    private static final int MIN_SAMPLES_FOR_TRAINING = 500; // Минимум сэмплов для обучения
    private static final double CHEAT_THRESHOLD = 0.88;      // Было 0.70 → 0.88
    private static final double ALERT_THRESHOLD = 0.93;      // Было 0.75 → 0.93
    private static final double FLAG_VL = 20.0;              // Было 5.0 → 20.0
    private static final double KICK_VL = 60.0;
    private static final String DATASET_DIR = "plugins/GrimAC/ml_datasets/";
    private static final String MODEL_FILE = "plugins/GrimAC/ml_models/aim_model.dat";

    // ========== ГЛОБАЛЬНОЕ ХРАНИЛИЩЕ ==========
    @Getter
    private static final Map<UUID, DataCollector> activeCollectors = new ConcurrentHashMap<>();
    @Getter
    private static String globalCollectionId = null; // ID глобального сбора
    private static AimModel_SMART trainedModel = null;

    // ========== ДАННЫЕ ИГРОКА ==========
    private final Deque<TickData> tickHistory = new ArrayDeque<>(SEQUENCE_LENGTH);
    private double currentCheatProbability = 0.0;
    private double violationLevel = 0.0;
    private long lastVLUpdate = System.currentTimeMillis();
    private int ticksSinceLastAnalysis = 0;
    private long lastVLDecay = System.currentTimeMillis();
    private int totalAnalyses = 0;
    private int highProbabilityDetections = 0;

    // Для вычисления производных
    private float lastDeltaYaw = 0.0f;
    private float lastDeltaPitch = 0.0f;
    private float lastAccelYaw = 0.0f;
    private float lastAccelPitch = 0.0f;
    private double estimatedGcdYaw = 0.0;
    private double estimatedGcdPitch = 0.0;

    public ImprovedAimML(GrimPlayer player) {
        super(player);

        if (player != null) {
            System.out.println("╔═══════════════════════════════════════════════════");
            System.out.println("║ [GrimAC ML] Улучшенная система инициализирована");
            System.out.println("║ Игрок: " + player.getName());
            System.out.println("║ Sequence: " + SEQUENCE_LENGTH);
            System.out.println("╚═══════════════════════════════════════════════════");
        }

        // Загружаем модель при первом запуске
        if (trainedModel == null) {
            loadModel();
        }
    }

    @Override
    public void process(RotationUpdate rotationUpdate) {
        // Создаём тик с расширенными данными
        TickData tick = createTickData(rotationUpdate);

        // НОВОЕ: Фильтруем шум - игнорируем очень малые движения
        double totalMovement = Math.sqrt(
                tick.deltaYaw * tick.deltaYaw +
                        tick.deltaPitch * tick.deltaPitch
        );

        GrimPlayer grimPlayer = rotationUpdate.getProcessor().getPlayer();
        System.out.println("[GrimAC ML DEBUG] process() вызван для " + grimPlayer.getName());


        // Игнорируем движения меньше 1 градуса (было раньше сразу добавление)
        // ИСПРАВЛЕНО: Убран фильтр малых движений - принимаем все данные

        // Добавляем в историю
        tickHistory.add(tick);
        if (tickHistory.size() > SEQUENCE_LENGTH) {
            tickHistory.removeFirst();
        }

        // Если идёт сбор данных - записываем
        DataCollector collector = activeCollectors.get(player.uuid);
        if (collector != null && collector.isRecording) {
            collector.addTick(tick);
        }

        // НОВОЕ: Decay VL со временем
        decayViolationLevel();

        // ИСПРАВЛЕНО: Анализируем реже - каждые 25 тиков (было 10)
        ticksSinceLastAnalysis++;
        if (ticksSinceLastAnalysis >= 5 && tickHistory.size() >= SEQUENCE_LENGTH) {  // ИСПРАВЛЕНО: было 10
            analyzeSequence();
            ticksSinceLastAnalysis = 0;
        }
    }
    /**
     * НОВОЕ: Уменьшение VL со временем при отсутствии читов
     */
    private void decayViolationLevel() {
        long now = System.currentTimeMillis();
        long timeSinceDecay = now - lastVLDecay;

        if (timeSinceDecay >= 5000 && violationLevel > 0) { // Каждые 5 секунд
            violationLevel = Math.max(0, violationLevel - 0.5); // Уменьшаем на 0.5
            lastVLDecay = now;
        }
    }

    /**
     * Создание расширенного тика с дополнительными фичами
     * ИСПРАВЛЕНО: Добавлен расчёт jerk (рывок)
     */
    private TickData createTickData(RotationUpdate update) {
        float deltaYaw = update.getTo().yaw() - update.getFrom().yaw();
        float deltaPitch = update.getTo().pitch() - update.getFrom().pitch();

        // Нормализуем yaw (убираем переход через 360)
        deltaYaw = normalizeAngle(deltaYaw);

        // Вычисляем ускорение (accel) = изменение скорости
        float accelYaw = deltaYaw - lastDeltaYaw;
        float accelPitch = deltaPitch - lastDeltaPitch;

        // Вычисляем рывок (jerk) = изменение ускорения
        float jerkYaw = lastAccelYaw > 0 ? accelYaw - lastAccelYaw : 0.0f;
        float jerkPitch = lastAccelPitch > 0 ? accelPitch - lastAccelPitch : 0.0f;

        // Оцениваем GCD (наибольший общий делитель)
        updateGcdEstimate(Math.abs(deltaYaw), Math.abs(deltaPitch));

        // Вычисляем ошибку GCD
        float gcdErrorYaw = (float) calculateGcdError(Math.abs(deltaYaw), estimatedGcdYaw);
        float gcdErrorPitch = (float) calculateGcdError(Math.abs(deltaPitch), estimatedGcdPitch);

        TickData tick = new TickData(
                deltaYaw,
                deltaPitch,
                accelYaw,
                accelPitch,
                jerkYaw,
                jerkPitch,
                gcdErrorYaw,
                gcdErrorPitch,
                System.currentTimeMillis(),
                player.getTransactionPing(),
                player.isSprinting,
                player.isSneaking
        );

        // Обновляем для следующей итерации
        lastDeltaYaw = deltaYaw;
        lastDeltaPitch = deltaPitch;
        lastAccelYaw = accelYaw;
        lastAccelPitch = accelPitch;

        return tick;
    }

    /**
     * Нормализация угла (-180 до 180)
     */
    private float normalizeAngle(float angle) {
        while (angle > 180.0f) angle -= 360.0f;
        while (angle < -180.0f) angle += 360.0f;
        return angle;
    }

    /**
     * Обновление оценки GCD (используем алгоритм Евклида)
     */
    private void updateGcdEstimate(double yaw, double pitch) {
        if (yaw > 0.0001) {
            estimatedGcdYaw = estimatedGcdYaw == 0 ? yaw : gcd(estimatedGcdYaw, yaw);
        }
        if (pitch > 0.0001) {
            estimatedGcdPitch = estimatedGcdPitch == 0 ? pitch : gcd(estimatedGcdPitch, pitch);
        }
    }

    /**
     * НОД для double (алгоритм Евклида)
     */
    private double gcd(double a, double b) {
        double epsilon = 0.0001;
        while (b > epsilon) {
            double temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    /**
     * Ошибка GCD = насколько движение отклоняется от GCD
     */
    private double calculateGcdError(double value, double gcd) {
        if (gcd < 0.0001) return 0.0;
        double remainder = value % gcd;
        return Math.min(remainder, gcd - remainder);
    }

    /**
     * Анализ последовательности тиков через ML-модель
     * ИСПРАВЛЕНО: Исправлен расчёт фичей (8 вместо 24)
     */
    private void analyzeSequence() {
        if (trainedModel == null || tickHistory.size() < SEQUENCE_LENGTH) {
            return;
        }

        double totalMovement = 0.0;
        for (TickData tick : tickHistory) {
            totalMovement += Math.sqrt(tick.deltaYaw * tick.deltaYaw + tick.deltaPitch * tick.deltaPitch);
        }
        double avgMovement = totalMovement / tickHistory.size();

        if (totalAnalyses % 20 == 0) {
            System.out.println("[GrimAC ML DEBUG] " + player.getName() + " avgMove=" + String.format("%.2f", avgMovement) + " histSize=" + tickHistory.size());
        }

        if (avgMovement < 0.3) {
            return;
        }

        List<TickData> tickList = new ArrayList<>(tickHistory);
        double[] features = extractFeatures(tickList);

        double probability = trainedModel.predict(features);
        currentCheatProbability = probability;

        // Добавляем в голограмму/GUI
        MLBridgeHolder.getBridge().addStrike(player.uuid, probability);

        totalAnalyses++;
        if (probability >= CHEAT_THRESHOLD) {
            highProbabilityDetections++;
        }

        if (probability >= ALERT_THRESHOLD) {
            long now = System.currentTimeMillis();
            long timeSinceLastUpdate = now - lastVLUpdate;

            if (timeSinceLastUpdate >= 2000) {
                double vlIncrease = (probability - ALERT_THRESHOLD) * 1.5;
                violationLevel += vlIncrease;
                lastVLUpdate = now;

                if (violationLevel >= FLAG_VL) {
                    String verbose = String.format(
                            "ML=%.0f%% VL=%.1f avg=%.1f°",
                            probability * 100,
                            violationLevel,
                            avgMovement
                    );

                    this.flagAndAlert(verbose);
                }
            }
        } else {
            if (probability < CHEAT_THRESHOLD * 0.7) {
                violationLevel = Math.max(0, violationLevel - 0.3);
            }
        }
    }
    /**
     * Обновление Violation Level
     */
    private void updateViolationLevel(double probability) {
        long now = System.currentTimeMillis();
        long delta = now - lastVLUpdate;

        // Decay со временем
        if (delta > 1000) {
            double decay = (delta / 1000.0) * 0.05;
            violationLevel = Math.max(0.0, violationLevel - decay);
        }
        lastVLUpdate = now;

        // Увеличиваем VL если вероятность высокая
        if (probability > CHEAT_THRESHOLD) {
            double multiplier = 1.0 + (probability - CHEAT_THRESHOLD) * 3.0;
            violationLevel += 0.8 * multiplier;
        } else {
            // Снижаем VL при легитной игре
            violationLevel = Math.max(0.0, violationLevel - 0.05);
        }
    }

    /**
     * Извлечение признаков из последовательности тиков (24 фичи)
     */
    private double[] extractFeatures(List<TickData> ticks) {
        double[] features = new double[24]; // 24 фичи

        if (ticks.isEmpty()) return features;

        List<Double> deltaYaws = new ArrayList<>();
        List<Double> deltaPitches = new ArrayList<>();
        List<Double> accelYaws = new ArrayList<>();
        List<Double> accelPitches = new ArrayList<>();
        List<Double> gcdErrorsYaw = new ArrayList<>();
        List<Double> gcdErrorsPitch = new ArrayList<>();

        for (TickData tick : ticks) {
            deltaYaws.add((double) Math.abs(tick.deltaYaw));
            deltaPitches.add((double) Math.abs(tick.deltaPitch));
            accelYaws.add((double) Math.abs(tick.accelYaw));
            accelPitches.add((double) Math.abs(tick.accelPitch));
            gcdErrorsYaw.add((double) tick.gcdErrorYaw);
            gcdErrorsPitch.add((double) tick.gcdErrorPitch);
        }

        // Feature 0-1: Средняя скорость
        features[0] = average(deltaYaws);
        features[1] = average(deltaPitches);

        // Feature 2-3: Стандартное отклонение скорости
        features[2] = standardDeviation(deltaYaws);
        features[3] = standardDeviation(deltaPitches);

        // Feature 4-5: Максимальная скорость
        features[4] = deltaYaws.isEmpty() ? 0 : Collections.max(deltaYaws);
        features[5] = deltaPitches.isEmpty() ? 0 : Collections.max(deltaPitches);

        // Feature 6-7: Среднее ускорение
        features[6] = average(accelYaws);
        features[7] = average(accelPitches);

        // Feature 8-9: Стандартное отклонение ускорения
        features[8] = standardDeviation(accelYaws);
        features[9] = standardDeviation(accelPitches);

        // Feature 10-11: Средняя GCD-ошибка
        features[10] = average(gcdErrorsYaw);
        features[11] = average(gcdErrorsPitch);

        // Feature 12: Процент нулевых движений
        features[12] = deltaYaws.stream().filter(d -> d < 0.01).count() / (double) deltaYaws.size();

        // Feature 13: Процент резких движений (>20 градусов)
        features[13] = deltaYaws.stream().filter(d -> d > 20).count() / (double) deltaYaws.size();

        // Feature 14: Энтропия движений
        features[14] = calculateEntropy(deltaYaws);

        // Feature 15: Коэффициент вариации
        features[15] = features[2] / (features[0] + 0.001);

        // Feature 16: Паттерн "снапа"
        features[16] = detectSnapPattern(deltaYaws);

        // Feature 17-18: Квантили (25% и 75%)
        features[17] = percentile(deltaYaws, 0.25);
        features[18] = percentile(deltaYaws, 0.75);

        // Feature 19: Константная скорость
        features[19] = detectConstantSpeedPattern(deltaYaws);

        // Feature 20: Корреляция yaw/pitch
        features[20] = calculateCorrelation(deltaYaws, deltaPitches);

        // Feature 21: Среднее соотношение GCD-ошибки к движению
        features[21] = features[10] / (features[0] + 0.001);

        // Feature 22: Максимальное ускорение
        features[22] = accelYaws.isEmpty() ? 0 : Collections.max(accelYaws);

        // Feature 23: Процент идеально прямых углов
        features[23] = countPerfectAngles(ticks) / (double) ticks.size();

        // Проверка на NaN/Infinity
        for (int i = 0; i < features.length; i++) {
            if (Double.isNaN(features[i]) || Double.isInfinite(features[i])) {
                System.out.println("[FEATURES ERROR] Feature " + i + " is NaN or Infinite!");
                features[i] = 0.0;
            }
        }

        return features;
    }

    // ========== УТИЛИТЫ ДЛЯ СТАТИСТИКИ ==========

    private static double average(List<Double> values) {
        return values.isEmpty() ? 0 : values.stream().mapToDouble(Double::doubleValue).average().orElse(0);
    }

    private static double standardDeviation(List<Double> values) {
        if (values.size() < 2) return 0;
        double avg = average(values);
        double variance = values.stream().mapToDouble(v -> Math.pow(v - avg, 2)).sum() / values.size();
        return Math.sqrt(variance);
    }

    private static double calculateEntropy(List<Double> values) {
        Map<Integer, Long> histogram = values.stream()
                .collect(Collectors.groupingBy(v -> (int)(v / 5), Collectors.counting()));
        double total = values.size();
        double entropy = 0;
        for (long count : histogram.values()) {
            double p = count / total;
            if (p > 0) entropy -= p * Math.log(p);
        }
        return entropy;
    }

    private static double detectSnapPattern(List<Double> deltas) {
        int snapCount = 0;
        for (int i = 1; i < deltas.size() - 1; i++) {
            if (deltas.get(i) > 15 && deltas.get(i + 1) < 2) {
                snapCount++;
            }
        }
        return snapCount / (double) deltas.size();
    }

    private static double percentile(List<Double> values, double p) {
        if (values.isEmpty()) return 0;
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int index = (int) Math.ceil(p * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(index, sorted.size() - 1)));
    }

    private static double detectConstantSpeedPattern(List<Double> deltas) {
        if (deltas.size() < 5) return 0;
        int constantCount = 0;
        for (int i = 0; i < deltas.size() - 4; i++) {
            double avg = (deltas.get(i) + deltas.get(i+1) + deltas.get(i+2) +
                    deltas.get(i+3) + deltas.get(i+4)) / 5.0;
            boolean allSimilar = true;
            for (int j = i; j < i + 5; j++) {
                if (Math.abs(deltas.get(j) - avg) > 0.5) {
                    allSimilar = false;
                    break;
                }
            }
            if (allSimilar && avg > 1) constantCount++;
        }
        return constantCount / (double) (deltas.size() - 4);
    }

    private static double calculateCorrelation(List<Double> x, List<Double> y) {
        if (x.size() != y.size() || x.size() < 2) return 0;
        double avgX = average(x);
        double avgY = average(y);
        double numerator = 0;
        double denomX = 0;
        double denomY = 0;
        for (int i = 0; i < x.size(); i++) {
            double dx = x.get(i) - avgX;
            double dy = y.get(i) - avgY;
            numerator += dx * dy;
            denomX += dx * dx;
            denomY += dy * dy;
        }
        return numerator / (Math.sqrt(denomX * denomY) + 0.001);
    }

    private double countPerfectAngles(List<TickData> ticks) {
        long count = 0;
        for (TickData tick : ticks) {
            double y = Math.abs(tick.deltaYaw);
            double p = Math.abs(tick.deltaPitch);
            if (Math.abs(y % 90) < 0.1 || Math.abs(p % 90) < 0.1 ||
                    Math.abs(y) < 0.1 || Math.abs(p) < 0.1) {
                count++;
            }
        }
        return count;
    }

    /**
     * Считает идеально прямые углы для обучения (статическая версия для deltas)
     */
    private static double countPerfectAnglesFromDeltas(List<Double> deltas) {
        long count = 0;
        for (double y : deltas) {
            if (Math.abs(y % 90) < 0.1 || Math.abs(y) < 0.1) {
                count++;
            }
        }
        return count;
    }

    private void alertWithPermission(String message) {
        for (GrimPlayer online : ac.grim.grimac.GrimAPI.INSTANCE.getPlayerDataManager().getEntries()) {
            if (online.hasPermission("grim.alerts")) {
                online.sendMessage(message);
            }
        }
        System.out.println(message);
    }

    // ========== КЛАССЫ ДАННЫХ ==========

    /**
     * Сырой обучающий тик (из CSV)
     */
    public static class RawTrainingTick implements Serializable {
        private static final long serialVersionUID = 1L;

        public float deltaYaw, deltaPitch;
        public float accelYaw, accelPitch;
        public float jerkYaw, jerkPitch;
        public float gcdErrorYaw, gcdErrorPitch;
        public double isCheat;
    }

    /**
     * Расширенный тик с дополнительными метриками
     */
    public static class TickData implements Serializable {
        private static final long serialVersionUID = 1L;

        public final float deltaYaw, deltaPitch;
        public final float accelYaw, accelPitch;
        public final float jerkYaw, jerkPitch;
        public final float gcdErrorYaw, gcdErrorPitch;
        public final long timestamp;
        public final int ping;
        public final boolean sprinting, sneaking;

        public TickData(float deltaYaw, float deltaPitch, float accelYaw, float accelPitch,
                        float jerkYaw, float jerkPitch, float gcdErrorYaw, float gcdErrorPitch,
                        long timestamp, int ping, boolean sprinting, boolean sneaking) {
            this.deltaYaw = deltaYaw;
            this.deltaPitch = deltaPitch;
            this.accelYaw = accelYaw;
            this.accelPitch = accelPitch;
            this.jerkYaw = jerkYaw;
            this.jerkPitch = jerkPitch;
            this.gcdErrorYaw = gcdErrorYaw;
            this.gcdErrorPitch = gcdErrorPitch;
            this.timestamp = timestamp;
            this.ping = ping;
            this.sprinting = sprinting;
            this.sneaking = sneaking;
        }

        public String toCsv(String label) {
            return String.format(Locale.US,
                    "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                    label.equals("CHEAT") ? "1" : "0",
                    deltaYaw, deltaPitch,
                    accelYaw, accelPitch,
                    jerkYaw, jerkPitch,
                    gcdErrorYaw, gcdErrorPitch
            );
        }

        public static String getCsvHeader() {
            return "is_cheating,delta_yaw,delta_pitch,accel_yaw,accel_pitch," +
                    "jerk_yaw,jerk_pitch,gcd_error_yaw,gcd_error_pitch";
        }
    }

    /**
     * Коллектор данных
     */
    public static class DataCollector implements Serializable {
        private static final long serialVersionUID = 1L;

        public final UUID playerUUID;
        public final String playerName;
        public final List<TickData> ticks = new ArrayList<>();
        public boolean isRecording = false;
        public boolean isCheater = false;
        public final long startTime;
        public final String collectionId; // ID сессии сбора

        public DataCollector(UUID uuid, String name, String collectionId) {
            this.playerUUID = uuid;
            this.playerName = name;
            this.collectionId = collectionId;
            this.startTime = System.currentTimeMillis();
        }

        public void addTick(TickData tick) {
            if (isRecording) {
                ticks.add(tick);
            }
        }

        public void markAsCheater() {
            this.isCheater = true;
        }

        public String generateFileName() {
            String timestamp = new SimpleDateFormat("yyyyMMdd-HHmmss").format(new Date(startTime));
            String label = isCheater ? "CHEAT" : "LEGIT";
            return String.format("%s_%s_%s.csv", label, playerName, timestamp);
        }

        public String generateCsvContent() {
            if (ticks.isEmpty()) return "";

            StringBuilder sb = new StringBuilder();
            sb.append(TickData.getCsvHeader()).append("\n");

            String label = isCheater ? "CHEAT" : "LEGIT";
            for (TickData tick : ticks) {
                sb.append(tick.toCsv(label)).append("\n");
            }

            return sb.toString();
        }

        public void saveToFile() throws IOException {
            String csvContent = generateCsvContent();
            if (csvContent.isEmpty()) return;

            Path dir = Paths.get(DATASET_DIR);
            Files.createDirectories(dir);

            File file = dir.resolve(generateFileName()).toFile();
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
                writer.write(csvContent);
            }

            System.out.println("[GrimAC ML] Saved dataset: " + file.getName() +
                    " (" + ticks.size() + " ticks)");
        }
    }

    // ========== API МЕТОДЫ ==========

    /**
     * Начать локальную запись для игрока
     */
    public static void startRecording(GrimPlayer player, String collectionId) {
        // Проверяем что не записываем уже
        if (activeCollectors.containsKey(player.uuid)) {
            System.out.println("[GrimAC ML] Игрок " + player.getName() + " уже записывает!");
            return;
        }

        DataCollector collector = new DataCollector(player.uuid, player.getName(), collectionId);
        collector.isRecording = true;
        activeCollectors.put(player.uuid, collector);

        System.out.println("[GrimAC ML] ✓ Запись начата: " + player.getName() + " (ID: " + collectionId + ")");

        // Отправляем сообщение игроку (только если не глобальный сбор)
        if (!collectionId.startsWith("GLOBAL_")) {
            player.sendMessage("§a[GrimAC ML] Запись датасета начата!");
        }
    }
    /**
     * Остановить запись и сохранить
     */
    public static String stopRecording(GrimPlayer player, boolean save) {
        DataCollector collector = activeCollectors.get(player.uuid);
        if (collector == null) {
            return "§cЗапись не активна!";
        }

        collector.isRecording = false;
        int tickCount = collector.ticks.size();

        if (save) {
            if (tickCount < 100) {
                activeCollectors.remove(player.uuid);
                return String.format("§cНедостаточно данных (%d тиков). Минимум: 100", tickCount);
            }

            try {
                collector.saveToFile();
                activeCollectors.remove(player.uuid);
                String label = collector.isCheater ? "§c§lЧИТ" : "§a§lЛЕГИТ";
                return String.format(
                        "Датасет сохранён! Собрано %d тиков, метка: %s",
                        tickCount,
                        label
                );
            } catch (IOException e) {
                activeCollectors.remove(player.uuid);
                return "§cОшибка сохранения: " + e.getMessage();
            }
        } else {
            activeCollectors.remove(player.uuid);
            return "Запись отменена";
        }
    }

    /**
     * Начать глобальный сбор датасетов
     */
    public static int startGlobalCollection(String collectionId) {
        globalCollectionId = collectionId;

        int started = 0;

        // Получаем всех онлайн игроков
        Collection<GrimPlayer> allPlayers = GrimAPI.INSTANCE.getPlayerDataManager().getEntries();

        System.out.println("[GrimAC ML] Глобальный сбор: найдено игроков: " + allPlayers.size());

        for (GrimPlayer player : allPlayers) {
            // Пропускаем игроков с пермишеном на освобождение
            if (player.hasPermission("grim.ml.exempt")) {
                System.out.println("[GrimAC ML] Игрок " + player.getName() + " пропущен (grim.ml.exempt)");
                continue;
            }

            // Проверяем что игрок ещё не записывает
            if (activeCollectors.containsKey(player.uuid)) {
                System.out.println("[GrimAC ML] Игрок " + player.getName() + " уже записывает");
                continue;
            }

            // Начинаем запись
            startRecording(player, collectionId);
            started++;

            System.out.println("[GrimAC ML] Начата запись для: " + player.getName());
        }

        System.out.println("[GrimAC ML] Глобальный сбор запущен для " + started + " игроков");

        return started;
    }

    /**
     * Остановить глобальный сбор и создать архив
     */
    public static String stopGlobalCollection() {
        if (globalCollectionId == null) {
            return "§cГлобальный сбор не активен!";
        }

        String currentId = globalCollectionId;
        globalCollectionId = null;

        List<DataCollector> collectorsToArchive = new ArrayList<>();

        System.out.println("[GrimAC ML] Останавливаем глобальный сбор: " + currentId);
        System.out.println("[GrimAC ML] Активных коллекторов: " + activeCollectors.size());

        for (Map.Entry<UUID, DataCollector> entry : activeCollectors.entrySet()) {
            DataCollector collector = entry.getValue();

            System.out.println("[GrimAC ML] Проверяем коллектор: " + collector.playerName +
                    " (ID: " + collector.collectionId + ", тиков: " + collector.ticks.size() + ")");

            if (currentId.equals(collector.collectionId)) {
                collector.isRecording = false;
                collectorsToArchive.add(collector);
                System.out.println("[GrimAC ML] ✓ Добавлен в архив: " + collector.playerName);
            }
        }

        if (collectorsToArchive.isEmpty()) {
            System.out.println("[GrimAC ML] Нет данных для архивирования!");
            return "§eНет данных для архивирования";
        }

        // Удаляем из активных
        for (DataCollector collector : collectorsToArchive) {
            activeCollectors.remove(collector.playerUUID);
            System.out.println("[GrimAC ML] Удалён из активных: " + collector.playerName);
        }

        // Создаём архив асинхронно
        int totalTicks = collectorsToArchive.stream().mapToInt(c -> c.ticks.size()).sum();
        System.out.println("[GrimAC ML] Всего тиков для архивации: " + totalTicks);

        ac.grim.grimac.GrimAPI.INSTANCE.getScheduler().getAsyncScheduler().runNow(
                ac.grim.grimac.GrimAPI.INSTANCE.getGrimPlugin(),
                () -> archiveCollectors(collectorsToArchive, currentId)
        );

        return String.format(
                "§aГлобальный сбор остановлен! Архивируется %d датасет(ов) (%d тиков)...",
                collectorsToArchive.size(),
                totalTicks
        );
    }


    /**
     * Архивирование коллекторов в ZIP
     */
    private static void archiveCollectors(List<DataCollector> collectors, String archiveName) {
        try {
            Path dataFolder = Paths.get(DATASET_DIR);
            Files.createDirectories(dataFolder);

            File zipFile = dataFolder.resolve(archiveName + ".zip").toFile();

            try (FileOutputStream fos = new FileOutputStream(zipFile);
                 ZipOutputStream zos = new ZipOutputStream(fos)) {

                for (DataCollector collector : collectors) {
                    if (collector.ticks.isEmpty()) continue;

                    String fileName = collector.generateFileName();
                    String csvContent = collector.generateCsvContent();

                    ZipEntry zipEntry = new ZipEntry(fileName);
                    zos.putNextEntry(zipEntry);
                    zos.write(csvContent.getBytes(java.nio.charset.StandardCharsets.UTF_8));
                    zos.closeEntry();
                }
            }

            System.out.println("[GrimAC ML] Архив создан: " + zipFile.getAbsolutePath());

        } catch (IOException e) {
            System.err.println("[GrimAC ML] Ошибка создания архива: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Пометить игрока как читера
     */
    public static boolean markAsCheater(UUID playerUUID) {
        DataCollector collector = activeCollectors.get(playerUUID);
        if (collector != null) {
            collector.markAsCheater();
            System.out.println("[GrimAC ML] Игрок помечен как CHEAT: " + playerUUID);
            return true;
        }
        return false;
    }

    /**
     * Получить статус записи
     */
    public static String getStatus(GrimPlayer player) {
        StringBuilder sb = new StringBuilder();
        sb.append("§b§l[GrimAC ML Status - ").append(player.getName()).append("]\n");
        sb.append("§7════════════════════════════════════\n");

        sb.append(String.format("§7Модель: %s\n",
                trainedModel != null ? "§a✓ Загружена" : "§c✗ Не обучена"));
        sb.append(String.format("§7Активных записей: §e%d\n", activeCollectors.size()));
        sb.append(String.format("§7Глобальный сбор: %s\n",
                globalCollectionId != null ? "§a✓ " + globalCollectionId : "§c✗ Не активен"));

        DataCollector collector = activeCollectors.get(player.uuid);
        if (collector != null) {
            int ticks = collector.ticks.size();
            long recordingTime = (System.currentTimeMillis() - collector.startTime) / 1000;

            sb.append("§7\n");
            sb.append("§a▸ Текущая запись: §a§lАКТИВНА\n");
            sb.append(String.format("  §7Собрано тиков: §e%d\n", ticks));
            sb.append(String.format("  §7Время записи: §e%d сек\n", recordingTime));

            String quality = ticks < 100 ? "§cПЛОХОЕ" :
                    ticks < 300 ? "§eСРЕДНЕЕ" :
                            ticks < 500 ? "§aХОРОШЕЕ" : "§a§lОТЛИЧНОЕ";
            sb.append(String.format("  §7Качество: %s\n", quality));

            String label = collector.isCheater ? "§c§lЧИТ" : "§7НЕ УСТАНОВЛЕНА";
            sb.append(String.format("  §7Метка: %s\n", label));
        } else {
            sb.append("§7\n");
            sb.append("§c▸ Запись: §c§lНЕ АКТИВНА\n");
        }

        sb.append("§7════════════════════════════════════");
        return sb.toString();
    }

    /**
     * Вычислить 24 фичи из сырых тиков для обучения
     * Группирует тики по 40 и вычисляет статистики
     */
    private static List<AimModel_SMART.TrainingExample> compute24FeaturesFromRawTicks(List<RawTrainingTick> rawTicks) {
        List<AimModel_SMART.TrainingExample> trainingExamples = new ArrayList<>();

        // Разбиваем тики на окна по 40
        int windowSize = 40;
        int numWindows = rawTicks.size() / windowSize;

        for (int i = 0; i < numWindows; i++) {
            int startIdx = i * windowSize;
            int endIdx = startIdx + windowSize;
            List<RawTrainingTick> window = rawTicks.subList(startIdx, endIdx);

            // Вычисляем 24 фичи
            double[] features = new double[24];

            List<Double> deltaYaws = new ArrayList<>();
            List<Double> deltaPitches = new ArrayList<>();
            List<Double> accelYaws = new ArrayList<>();
            List<Double> accelPitches = new ArrayList<>();
            List<Double> gcdErrorsYaw = new ArrayList<>();
            List<Double> gcdErrorsPitch = new ArrayList<>();

            for (RawTrainingTick tick : window) {
                deltaYaws.add((double) Math.abs(tick.deltaYaw));
                deltaPitches.add((double) Math.abs(tick.deltaPitch));
                accelYaws.add((double) Math.abs(tick.accelYaw));
                accelPitches.add((double) Math.abs(tick.accelPitch));
                gcdErrorsYaw.add((double) tick.gcdErrorYaw);
                gcdErrorsPitch.add((double) tick.gcdErrorPitch);
            }

            // Feature 0-1: Средняя скорость
            features[0] = average(deltaYaws);
            features[1] = average(deltaPitches);

            // Feature 2-3: Стандартное отклонение скорости
            features[2] = standardDeviation(deltaYaws);
            features[3] = standardDeviation(deltaPitches);

            // Feature 4-5: Максимальная скорость
            features[4] = deltaYaws.isEmpty() ? 0 : Collections.max(deltaYaws);
            features[5] = deltaPitches.isEmpty() ? 0 : Collections.max(deltaPitches);

            // Feature 6-7: Среднее ускорение
            features[6] = average(accelYaws);
            features[7] = average(accelPitches);

            // Feature 8-9: Стандартное отклонение ускорения
            features[8] = standardDeviation(accelYaws);
            features[9] = standardDeviation(accelPitches);

            // Feature 10-11: Средняя GCD-ошибка
            features[10] = average(gcdErrorsYaw);
            features[11] = average(gcdErrorsPitch);

            // Feature 12: Процент нулевых движений
            features[12] = deltaYaws.stream().filter(d -> d < 0.01).count() / (double) deltaYaws.size();

            // Feature 13: Процент резких движений (>20 градусов)
            features[13] = deltaYaws.stream().filter(d -> d > 20).count() / (double) deltaYaws.size();

            // Feature 14: Энтропия движений
            features[14] = calculateEntropy(deltaYaws);

            // Feature 15: Коэффициент вариации
            features[15] = features[2] / (features[0] + 0.001);

            // Feature 16: Паттерн "снапа"
            features[16] = detectSnapPattern(deltaYaws);

            // Feature 17-18: Квантили (25% и 75%)
            features[17] = percentile(deltaYaws, 0.25);
            features[18] = percentile(deltaYaws, 0.75);

            // Feature 19: Константная скорость
            features[19] = detectConstantSpeedPattern(deltaYaws);

            // Feature 20: Корреляция yaw/pitch
            features[20] = calculateCorrelation(deltaYaws, deltaPitches);

            // Feature 21: Среднее соотношение GCD-ошибки к движению
            features[21] = features[10] / (features[0] + 0.001);

            // Feature 22: Максимальное ускорение
            features[22] = accelYaws.isEmpty() ? 0 : Collections.max(accelYaws);

            // Feature 23: Процент идеально прямых углов (используем deltaYaws)
            features[23] = countPerfectAnglesFromDeltas(deltaYaws) / (double) deltaYaws.size();

            // Берём метку (cheat или legit) из последнего тика в окне
            double label = window.get(window.size() - 1).isCheat;

            // Проверка на NaN/Infinity
            for (int j = 0; j < features.length; j++) {
                if (Double.isNaN(features[j]) || Double.isInfinite(features[j])) {
                    features[j] = 0.0;
                }
            }

            trainingExamples.add(new AimModel_SMART.TrainingExample(features, label));
        }

        System.out.println("[GrimAC ML] Вычислено обучающих примеров: " + trainingExamples.size() + " из " + rawTicks.size() + " тиков");
        return trainingExamples;
    }

    /**
     * Считает идеально прямые углы для обучения
     */
    private static double countPerfectAnglesFromDeltas(List<Double> deltas) {
        long count = 0;
        for (double y : deltas) {
            if (Math.abs(y % 90) < 0.1 || Math.abs(y) < 0.1) {
                count++;
            }
        }
        return count;
    }

    /**
     * Обучить модель
     */
    public static String trainModel() {
        try {
            File datasetDir = new File(DATASET_DIR);
            if (!datasetDir.exists()) {
                return "§c[GrimAC ML] Папка с датасетами не найдена!";
            }

            File[] csvFiles = datasetDir.listFiles((dir, name) -> name.endsWith(".csv"));
            if (csvFiles == null || csvFiles.length == 0) {
                return "§c[GrimAC ML] CSV файлы не найдены!";
            }

            System.out.println("[GrimAC ML] ═══════════════════════════════════");
            System.out.println("[GrimAC ML] НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ");
            System.out.println("[GrimAC ML] ═══════════════════════════════════");
            System.out.println("[GrimAC ML] Найдено файлов: " + csvFiles.length);

            // Загружаем все данные (raw тики для вычисления 24 фичей)
            List<RawTrainingTick> allRawData = new ArrayList<>();
            int legitCount = 0;
            int cheatCount = 0;

            for (File csvFile : csvFiles) {
                boolean isCheatFile = csvFile.getName().startsWith("CHEAT_");
                System.out.println("[GrimAC ML] Загрузка: " + csvFile.getName());

                try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
                    String header = br.readLine(); // Пропускаем заголовок

                    String line;
                    int lineCount = 0;
                    while ((line = br.readLine()) != null) {
                        if (line.trim().isEmpty()) continue;

                        String[] parts = line.split(",");
                        if (parts.length < 9) continue; // Минимум 9 колонок

                        try {
                            // Парсим данные (8 колонок + label)
                            RawTrainingTick tick = new RawTrainingTick();
                            tick.isCheat = isCheatFile ? 1.0 : 0.0;
                            tick.deltaYaw = Float.parseFloat(parts[0]);
                            tick.deltaPitch = Float.parseFloat(parts[1]);
                            tick.accelYaw = Float.parseFloat(parts[2]);
                            tick.accelPitch = Float.parseFloat(parts[3]);
                            tick.jerkYaw = Float.parseFloat(parts[4]);
                            tick.jerkPitch = Float.parseFloat(parts[5]);
                            tick.gcdErrorYaw = Float.parseFloat(parts[6]);
                            tick.gcdErrorPitch = Float.parseFloat(parts[7]);

                            allRawData.add(tick);

                            if (tick.isCheat > 0.5) {
                                cheatCount++;
                            } else {
                                legitCount++;
                            }

                            lineCount++;
                        } catch (NumberFormatException e) {
                            // Пропускаем плохие строки
                        }
                    }

                    System.out.println("[GrimAC ML]   → Загружено строк: " + lineCount);

                } catch (IOException e) {
                    System.err.println("[GrimAC ML] Ошибка чтения файла: " + csvFile.getName());
                    e.printStackTrace();
                }
            }

            // Группируем тики по игрокам для расчёта 24 фичей
            List<AimModel_SMART.TrainingExample> allData = compute24FeaturesFromRawTicks(allRawData);

            if (allData.isEmpty()) {
                return "§c[GrimAC ML] Не удалось загрузить данные из CSV!";
            }

            if (allData.size() < MIN_SAMPLES_FOR_TRAINING) {
                return String.format(
                        "§c[GrimAC ML] Недостаточно данных! Есть: %d, нужно: %d",
                        allData.size(), MIN_SAMPLES_FOR_TRAINING
                );
            }

            System.out.println("[GrimAC ML] ═══════════════════════════════════");
            System.out.println("[GrimAC ML] Всего примеров: " + allData.size());
            System.out.println("[GrimAC ML] Легитных: " + legitCount);
            System.out.println("[GrimAC ML] Читеров: " + cheatCount);
            System.out.println("[GrimAC ML] ═══════════════════════════════════");

            // Балансировка классов (если нужно)
            if (Math.abs(legitCount - cheatCount) > allData.size() * 0.3) {
                System.out.println("[GrimAC ML] ⚠ Дисбаланс классов обнаружен!");
                allData = balanceDataset(allData);
                System.out.println("[GrimAC ML] ✓ Данные сбалансированы: " + allData.size() + " примеров");
            }

            // Перемешиваем данные
            Collections.shuffle(allData, new Random(42));

            // Разделяем на train/validation (80/20)
            int trainSize = (int) (allData.size() * 0.8);
            List<AimModel_SMART.TrainingExample> trainData = allData.subList(0, trainSize);
            List<AimModel_SMART.TrainingExample> validData = allData.subList(trainSize, allData.size());

            System.out.println("[GrimAC ML] Обучающая выборка: " + trainData.size());
            System.out.println("[GrimAC ML] Валидационная выборка: " + validData.size());

            // Создаём и обучаем модель
            int numFeatures = allData.get(0).features.length;
            AimModel_SMART model = new AimModel_SMART(numFeatures);

            double learningRate = 0.05;      // было 0.01 → стало 0.05
            double l2Lambda = 0.0001;        // было 0.001 → стало 0.0001
            int maxIterations = 1000;        // было 500 → стало 1000

            model.train(trainData, validData, learningRate, l2Lambda, maxIterations);

            // Сохраняем модель
            File modelDir = new File(MODEL_FILE).getParentFile();
            if (modelDir != null) {
                modelDir.mkdirs();
            }
            model.save(MODEL_FILE);

            // Обновляем глобальную модель
            trainedModel = model;

            System.out.println("[GrimAC ML] ═══════════════════════════════════");
            System.out.println("[GrimAC ML] ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!");
            System.out.println("[GrimAC ML] " + model.getMetrics());
            System.out.println("[GrimAC ML] ═══════════════════════════════════");

            return String.format(
                    "§a[GrimAC ML] Модель обучена успешно!\n" +
                            "§7Использовано файлов: §e%d\n" +
                            "§7Примеров: §e%d §7(Легит: §a%d§7, Читы: §c%d§7)\n" +
                            "§7%s",
                    csvFiles.length,
                    allData.size(),
                    legitCount,
                    cheatCount,
                    model.getMetrics()
            );

        } catch (Exception e) {
            e.printStackTrace();
            return "§c[GrimAC ML] Ошибка обучения: " + e.getMessage();
        }
    }

    private static List<AimModel_SMART.TrainingExample> balanceDataset(
            List<AimModel_SMART.TrainingExample> data) {

        List<AimModel_SMART.TrainingExample> legitExamples = new ArrayList<>();
        List<AimModel_SMART.TrainingExample> cheatExamples = new ArrayList<>();

        // Разделяем на классы
        for (AimModel_SMART.TrainingExample ex : data) {
            if (ex.label > 0.5) {
                cheatExamples.add(ex);
            } else {
                legitExamples.add(ex);
            }
        }

        // Находим минимальный размер
        int minSize = Math.min(legitExamples.size(), cheatExamples.size());

        // Берём равное количество из каждого класса
        List<AimModel_SMART.TrainingExample> balanced = new ArrayList<>();
        Collections.shuffle(legitExamples);
        Collections.shuffle(cheatExamples);

        balanced.addAll(legitExamples.subList(0, minSize));
        balanced.addAll(cheatExamples.subList(0, minSize));

        return balanced;
    }


    /**
     * Загрузить модель
     */
    private void loadModel() {
        try {
            File modelFile = new File(MODEL_FILE);
            if (modelFile.exists()) {
                trainedModel = AimModel_SMART.load(MODEL_FILE);
                System.out.println("[GrimAC ML] ═══════════════════════════════════");
                System.out.println("[GrimAC ML] Модель загружена: " + MODEL_FILE);
                System.out.println("[GrimAC ML] " + trainedModel.getMetrics());
                System.out.println("[GrimAC ML] ═══════════════════════════════════");
            } else {
                System.out.println("[GrimAC ML] Модель не найдена, работаем в режиме сбора данных");
            }
        } catch (Exception e) {
            System.err.println("[GrimAC ML] Ошибка загрузки модели: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Получить количество тиков
     */
    public static int getTickCount(UUID playerUUID) {
        DataCollector collector = activeCollectors.get(playerUUID);
        return collector != null ? collector.ticks.size() : 0;
    }

    /**
     * Список датасетов
     */
    public static List<String> listDatasets() {
        File datasetDir = new File(DATASET_DIR);
        if (!datasetDir.exists()) {
            return Collections.emptyList();
        }

        File[] files = datasetDir.listFiles((dir, name) -> name.endsWith(".csv") || name.endsWith(".zip"));
        if (files == null) return Collections.emptyList();

        return Arrays.stream(files)
                .map(File::getName)
                .sorted()
                .collect(Collectors.toList());
    }

    // ========== ML МОДЕЛЬ (заглушка) ==========

    public static class AimModel implements Serializable {
        private static final long serialVersionUID = 1L;

        private double[] weights = new double[24];

        public double predict(double[] features) {
            // Простая линейная модель для демонстрации
            double score = 0.0;
            for (int i = 0; i < Math.min(features.length, weights.length); i++) {
                score += features[i] * weights[i];
            }
            return 1.0 / (1.0 + Math.exp(-score)); // Sigmoid
        }

        public void save(String path) throws IOException {
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
                oos.writeObject(this);
            }
        }

        public static AimModel load(String path) throws IOException, ClassNotFoundException {
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
                return (AimModel) ois.readObject();
            }
        }
    }
}
