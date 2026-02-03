package ac.grim.grimac.platform.bukkit.player;

import org.bukkit.Bukkit;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 * Система наказаний для ML детекции
 *
 * PACKAGE: ac.grim.grimac.platform.bukkit.player
 * FOR: Purpur 1.21.1
 *
 * @author ImprovedAImML Team
 * @version 1.0
 */
public class PenaltySystem {

    private final JavaPlugin plugin;
    private final Logger logger;

    // VL tracking
    private final Map<UUID, Integer> violationLevels = new ConcurrentHashMap<>();
    private final Map<UUID, Long> lastPunishmentTime = new ConcurrentHashMap<>();

    // Penalty commands
    private final Map<Integer, String> penaltyCommands = new ConcurrentHashMap<>();

    // History
    private final LinkedList<PenaltyRecord> penaltyHistory = new LinkedList<>();
    private static final int MAX_HISTORY_SIZE = 50;

    // Settings
    private double minProbability = 0.85;
    private long punishmentCooldown = 5000;
    private boolean vlDecayEnabled = true;
    private int vlDecayAmount = 1;
    private int vlDecayIntervalSeconds = 60;

    // Decay task
    private int decayTaskId = -1;

    public PenaltySystem(JavaPlugin plugin) {
        this.plugin = plugin;
        this.logger = plugin.getLogger();

        // Default penalties
        setupDefaultPenalties();

        // Start decay
        startDecayTask();
    }

    private void setupDefaultPenalties() {
        penaltyCommands.put(5, "kick {PLAYER} §c[ML] Подозрительная активность §7[VL: {VL}]");
        penaltyCommands.put(10, "tempban {PLAYER} 1h §c[ML] Читы обнаружены §7[Prob: {PROBABILITY}]");
        penaltyCommands.put(15, "ban {PLAYER} §c[ML] Использование читов");
    }

    public void handleViolation(Player player, double probability, double buffer) {
        UUID uuid = player.getUniqueId();

        if (probability < minProbability) {
            return;
        }

        if (isOnCooldown(uuid)) {
            return;
        }

        int newVL = incrementVL(uuid);

        logger.info(String.format(
                "[Penalty] %s - VL: %d | Prob: %.2f | Buffer: %.1f",
                player.getName(), newVL, probability, buffer
        ));

        String command = findApplicableCommand(newVL);
        if (command != null) {
            executePenalty(player, command, newVL, probability, buffer);
            lastPunishmentTime.put(uuid, System.currentTimeMillis());
        }
    }

    private boolean isOnCooldown(UUID uuid) {
        Long lastTime = lastPunishmentTime.get(uuid);
        if (lastTime == null) {
            return false;
        }

        long elapsed = System.currentTimeMillis() - lastTime;
        return elapsed < punishmentCooldown;
    }

    public int incrementVL(UUID uuid) {
        return violationLevels.merge(uuid, 1, Integer::sum);
    }

    public int getVL(UUID uuid) {
        return violationLevels.getOrDefault(uuid, 0);
    }

    public void resetVL(UUID uuid) {
        violationLevels.remove(uuid);
    }

    public void decreaseVL(UUID uuid, int amount) {
        violationLevels.computeIfPresent(uuid, (k, v) -> {
            int newVL = v - amount;
            return newVL <= 0 ? null : newVL;
        });
    }

    private String findApplicableCommand(int vl) {
        if (penaltyCommands.isEmpty()) {
            return null;
        }

        if (penaltyCommands.containsKey(vl)) {
            return penaltyCommands.get(vl);
        }

        int maxThreshold = -1;
        int applicableThreshold = -1;

        for (int threshold : penaltyCommands.keySet()) {
            if (threshold > maxThreshold) {
                maxThreshold = threshold;
            }

            if (threshold <= vl && threshold > applicableThreshold) {
                applicableThreshold = threshold;
            }
        }

        if (applicableThreshold == -1 && vl > maxThreshold) {
            return penaltyCommands.get(maxThreshold);
        }

        return applicableThreshold > 0 ? penaltyCommands.get(applicableThreshold) : null;
    }

    private void executePenalty(Player player, String command, int vl, double probability, double buffer) {
        String processedCommand = command
                .replace("{PLAYER}", player.getName())
                .replace("{VL}", String.valueOf(vl))
                .replace("{PROBABILITY}", String.format("%.2f", probability))
                .replace("{BUFFER}", String.format("%.1f", buffer));

        ActionType actionType = ActionType.fromCommand(processedCommand);

        Bukkit.getScheduler().runTask(plugin, () -> {
            boolean success = Bukkit.dispatchCommand(Bukkit.getConsoleSender(), processedCommand);

            if (success) {
                logger.info(String.format(
                        "[Penalty] Executed %s for %s (VL: %d)",
                        actionType, player.getName(), vl
                ));
            } else {
                logger.warning(String.format(
                        "[Penalty] Failed to execute %s for %s",
                        actionType, player.getName()
                ));
            }
        });

        addToHistory(new PenaltyRecord(
                player.getName(),
                player.getUniqueId(),
                actionType,
                vl,
                probability,
                buffer,
                processedCommand,
                System.currentTimeMillis()
        ));
    }

    private void addToHistory(PenaltyRecord record) {
        synchronized (penaltyHistory) {
            penaltyHistory.addFirst(record);

            while (penaltyHistory.size() > MAX_HISTORY_SIZE) {
                penaltyHistory.removeLast();
            }
        }
    }

    public List<PenaltyRecord> getHistory() {
        synchronized (penaltyHistory) {
            return Collections.unmodifiableList(new ArrayList<>(penaltyHistory));
        }
    }

    private void startDecayTask() {
        if (!vlDecayEnabled) {
            return;
        }

        long intervalTicks = vlDecayIntervalSeconds * 20L;

        decayTaskId = Bukkit.getScheduler().runTaskTimer(plugin, () -> {
            for (UUID uuid : violationLevels.keySet()) {
                decreaseVL(uuid, vlDecayAmount);
            }
        }, intervalTicks, intervalTicks).getTaskId();

        logger.info("[Penalty] VL decay started (interval: " + vlDecayIntervalSeconds + "s)");
    }

    private void stopDecayTask() {
        if (decayTaskId != -1) {
            Bukkit.getScheduler().cancelTask(decayTaskId);
            decayTaskId = -1;
        }
    }

    public void clearAll() {
        violationLevels.clear();
        lastPunishmentTime.clear();

        synchronized (penaltyHistory) {
            penaltyHistory.clear();
        }
    }

    public void shutdown() {
        stopDecayTask();
        clearAll();
    }

    // ========== SETTERS ==========

    public void setMinProbability(double minProbability) {
        this.minProbability = minProbability;
    }

    public void setPunishmentCooldown(long cooldownMs) {
        this.punishmentCooldown = cooldownMs;
    }

    public void setVLDecayEnabled(boolean enabled) {
        if (this.vlDecayEnabled != enabled) {
            this.vlDecayEnabled = enabled;

            if (enabled) {
                startDecayTask();
            } else {
                stopDecayTask();
            }
        }
    }

    public void setVLDecayAmount(int amount) {
        this.vlDecayAmount = amount;
    }

    public void setVLDecayInterval(int seconds) {
        this.vlDecayIntervalSeconds = seconds;

        if (vlDecayEnabled) {
            stopDecayTask();
            startDecayTask();
        }
    }

    public void setPenaltyCommands(Map<Integer, String> commands) {
        penaltyCommands.clear();
        penaltyCommands.putAll(commands);
    }

    // ========== GETTERS ==========

    public double getMinProbability() {
        return minProbability;
    }

    public long getPunishmentCooldown() {
        return punishmentCooldown;
    }

    public boolean isVLDecayEnabled() {
        return vlDecayEnabled;
    }

    public int getVLDecayAmount() {
        return vlDecayAmount;
    }

    public int getVLDecayInterval() {
        return vlDecayIntervalSeconds;
    }

    public Map<Integer, String> getPenaltyCommands() {
        return Collections.unmodifiableMap(penaltyCommands);
    }

    /**
     * Action Type enum
     */
    public enum ActionType {
        KICK,
        BAN,
        TEMPBAN,
        CUSTOM,
        UNKNOWN;

        public static ActionType fromCommand(String command) {
            String lower = command.toLowerCase();

            if (lower.startsWith("kick ")) {
                return KICK;
            } else if (lower.startsWith("ban ")) {
                return BAN;
            } else if (lower.startsWith("tempban ")) {
                return TEMPBAN;
            } else if (lower.startsWith("mute ") || lower.startsWith("warn ")) {
                return CUSTOM;
            }

            return UNKNOWN;
        }
    }

    /**
     * Penalty Record
     */
    public static class PenaltyRecord {
        public final String playerName;
        public final UUID playerUUID;
        public final ActionType actionType;
        public final int violationLevel;
        public final double probability;
        public final double buffer;
        public final String command;
        public final long timestamp;

        public PenaltyRecord(String playerName, UUID playerUUID, ActionType actionType,
                             int vl, double probability, double buffer,
                             String command, long timestamp) {
            this.playerName = playerName;
            this.playerUUID = playerUUID;
            this.actionType = actionType;
            this.violationLevel = vl;
            this.probability = probability;
            this.buffer = buffer;
            this.command = command;
            this.timestamp = timestamp;
        }

        public String getFormattedTimestamp() {
            return new java.text.SimpleDateFormat("HH:mm:ss")
                    .format(new Date(timestamp));
        }
    }
}
