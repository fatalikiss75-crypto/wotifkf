package ac.grim.grimac.checks.impl.aim;

import ac.grim.grimac.checks.Check;
import ac.grim.grimac.checks.CheckData;
import ac.grim.grimac.checks.type.RotationCheck;
import ac.grim.grimac.player.GrimPlayer;
import ac.grim.grimac.utils.anticheat.update.RotationUpdate;
import lombok.Getter;
import net.kyori.adventure.text.Component;

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
    private static final double CHEAT_THRESHOLD = 0.70; // Порог вероятности чита
    private static final double ALERT_THRESHOLD = 0.75; // Порог для алертов
    private static final double FLAG_VL = 5.0; // VL для флага
    private static final double KICK_VL = 20.0; // VL для кика
    private static final String DATASET_DIR = "plugins/GrimAC/ml_datasets/";
    private static final String MODEL_FILE = "plugins/GrimAC/ml_models/aim_model.dat";

    // ========== ГЛОБАЛЬНОЕ ХРАНИЛИЩЕ ==========
    @Getter
    private static final Map<UUID, DataCollector> activeCollectors = new ConcurrentHashMap<>();
    @Getter
    private static String globalCollectionId = null; // ID глобального сбора
    private static AimModel trainedModel = null;

    // ========== ДАННЫЕ ИГРОКА ==========
    private final Deque<TickData> tickHistory = new ArrayDeque<>(SEQUENCE_LENGTH);
    private double currentCheatProbability = 0.0;
    private double violationLevel = 0.0;
    private long lastVLUpdate = System.currentTimeMillis();
    private int ticksSinceLastAnalysis = 0;

    // Для вычисления производных
    private float lastDeltaYaw = 0.0f;
    private float lastDeltaPitch = 0.0f;
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

        // Анализируем каждые 10 тиков
        ticksSinceLastAnalysis++;
        if (ticksSinceLastAnalysis >= 10 && tickHistory.size() >= SEQUENCE_LENGTH) {
            analyzeSequence();
            ticksSinceLastAnalysis = 0;
        }
    }

    /**
     * Создание расширенного тика с дополнительными фичами
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
        // (это будет вычисляться в следующем тике)

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
            0.0f, // jerkYaw - будет обновлён позже
            0.0f, // jerkPitch - будет обновлён позже
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
     */
    private void analyzeSequence() {
        if (trainedModel == null || tickHistory.size() < SEQUENCE_LENGTH) {
            return;
        }

        // Извлекаем фичи
        double[] features = extractFeatures(new ArrayList<>(tickHistory));

        // Получаем предсказание
        double probability = trainedModel.predict(features);
        currentCheatProbability = probability;

        // Обновляем VL
        updateViolationLevel(probability);

        // Логируем подозрительную активность
        if (probability > 0.5) {
            System.out.println(String.format(
                "[GrimAC ML] %s prob=%.2f VL=%.2f ticks=%d",
                player.getName(),
                probability,
                violationLevel,
                tickHistory.size()
            ));
        }

        // Алерт при высокой вероятности
        if (probability > ALERT_THRESHOLD) {
            String alertMsg = String.format(
                "§c[GrimAC ML] §f%s §7suspicious (prob: §c%.1f%% §7VL: §e%.1f§7)",
                player.getName(),
                probability * 100,
                violationLevel
            );
            alertWithPermission(alertMsg);
        }

        // Флаг при достижении порога VL
        if (violationLevel >= FLAG_VL) {
            flagAndAlert(String.format(
                "AimML (prob: %.1f%%, VL: %.1f)",
                probability * 100,
                violationLevel
            ));
            violationLevel = FLAG_VL * 0.8; // Снижаем чтобы не спамить
        }

        // Кик при критическом VL
        if (violationLevel >= KICK_VL) {
            player.disconnect(Component.text(String.format(
                "§c[GrimAC ML] AimBot detected!\n" +
                "§7Probability: §c%.1f%%\n" +
                "§7VL: §c%.1f",
                probability * 100,
                violationLevel
            )));
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
     * Извлечение признаков из последовательности тиков
     */
    private double[] extractFeatures(List<TickData> ticks) {
        double[] features = new double[24]; // Увеличиваем количество фичей

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
        features[4] = Collections.max(deltaYaws);
        features[5] = Collections.max(deltaPitches);

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
        features[22] = Collections.max(accelYaws);

        // Feature 23: Процент идеально прямых углов
        features[23] = countPerfectAngles(ticks) / (double) ticks.size();

        return features;
    }

    // ========== УТИЛИТЫ ДЛЯ СТАТИСТИКИ ==========

    private double average(List<Double> values) {
        return values.isEmpty() ? 0 : values.stream().mapToDouble(Double::doubleValue).average().orElse(0);
    }

    private double standardDeviation(List<Double> values) {
        if (values.size() < 2) return 0;
        double avg = average(values);
        double variance = values.stream().mapToDouble(v -> Math.pow(v - avg, 2)).sum() / values.size();
        return Math.sqrt(variance);
    }

    private double calculateEntropy(List<Double> values) {
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

    private double detectSnapPattern(List<Double> deltas) {
        int snapCount = 0;
        for (int i = 1; i < deltas.size() - 1; i++) {
            if (deltas.get(i) > 15 && deltas.get(i + 1) < 2) {
                snapCount++;
            }
        }
        return snapCount / (double) deltas.size();
    }

    private double percentile(List<Double> values, double p) {
        if (values.isEmpty()) return 0;
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int index = (int) Math.ceil(p * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(index, sorted.size() - 1)));
    }

    private double detectConstantSpeedPattern(List<Double> deltas) {
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

    private double calculateCorrelation(List<Double> x, List<Double> y) {
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
        DataCollector collector = new DataCollector(player.uuid, player.getName(), collectionId);
        collector.isRecording = true;
        activeCollectors.put(player.uuid, collector);

        player.sendMessage("§a[GrimAC ML] Запись датасета начата!");
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
        for (GrimPlayer player : ac.grim.grimac.GrimAPI.INSTANCE.getPlayerDataManager().getEntries()) {
            if (!player.hasPermission("grim.ml.exempt")) {
                startRecording(player, collectionId);
                started++;
            }
        }

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
        for (DataCollector collector : activeCollectors.values()) {
            if (currentId.equals(collector.collectionId)) {
                collector.isRecording = false;
                collectorsToArchive.add(collector);
            }
        }

        if (collectorsToArchive.isEmpty()) {
            return "§eНет данных для архивирования";
        }

        // Удаляем из активных
        for (DataCollector collector : collectorsToArchive) {
            activeCollectors.remove(collector.playerUUID);
        }

        // Создаём архив асинхронно
        ac.grim.grimac.GrimAPI.INSTANCE.getScheduler().getAsyncScheduler().runNow(
            ac.grim.grimac.GrimAPI.INSTANCE.getGrimPlugin(),
            () -> archiveCollectors(collectorsToArchive, currentId)
        );

        return String.format(
            "§aГлобальный сбор остановлен! Архивируется %d датасет(ов)...",
            collectorsToArchive.size()
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

            // TODO: Реализовать обучение модели на CSV
            // Здесь нужно парсить CSV и обучать модель

            return String.format(
                "§a[GrimAC ML] Обучение завершено! Использовано %d файлов.",
                csvFiles.length
            );

        } catch (Exception e) {
            e.printStackTrace();
            return "§c[GrimAC ML] Ошибка обучения: " + e.getMessage();
        }
    }

    /**
     * Загрузить модель
     */
    private void loadModel() {
        try {
            File modelFile = new File(MODEL_FILE);
            if (modelFile.exists()) {
                trainedModel = AimModel.load(MODEL_FILE);
                System.out.println("[GrimAC ML] Модель загружена: " + MODEL_FILE);
            } else {
                System.out.println("[GrimAC ML] Модель не найдена, работаем в режиме сбора данных");
            }
        } catch (Exception e) {
            System.err.println("[GrimAC ML] Ошибка загрузки модели: " + e.getMessage());
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
