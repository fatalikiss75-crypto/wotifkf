package ac.grim.grimac.platform.bukkit;

import ac.grim.grimac.checks.impl.aim.DatasetBalanceChecker;
import ac.grim.grimac.platform.api.sender.Sender;
import ac.grim.grimac.platform.bukkit.player.MLMenuGUI;
import ac.grim.grimac.platform.bukkit.player.MLConfig;
import ac.grim.grimac.platform.bukkit.player.PenaltySystem;
import org.bukkit.Bukkit;
import org.bukkit.entity.Player;
import org.incendo.cloud.CommandManager;
import org.incendo.cloud.parser.standard.IntegerParser;
import org.incendo.cloud.parser.standard.StringParser;

/**
 * УЛУЧШЕННЫЙ регистратор ML команд для Bukkit модуля
 *
 * ИНТЕГРАЦИЯ:
 * - MLConfig (конфигурация)
 * - PenaltySystem (система наказаний)
 * - MLMenuGUI_Improved (оптимизированное GUI)
 * - DatasetBalanceChecker_Improved (анализ датасетов)
 *
 * @author ImprovedAImML Team
 * @version 2.0 (Purpur 1.21.1)
 */
public final class MLCommandRegistrar {

    // Singleton instances (инициализируются в GrimACBukkitLoaderPlugin)
    private static MLConfig config;
    private static PenaltySystem penaltySystem;

    /**
     * Установить instances (вызывается из GrimACBukkitLoaderPlugin.onEnable)
     */
    public static void setInstances(MLConfig mlConfig, PenaltySystem penalty) {
        config = mlConfig;
        penaltySystem = penalty;
    }

    public static void register(CommandManager<Sender> manager) {

        // ========== /tushpAcList - Открыть GUI ==========
        manager.command(
                manager.commandBuilder("tushpAcList")
                        .permission("grim.ml.list")
                        .handler(context -> {
                            Sender sender = context.sender();

                            if (!sender.isPlayer()) {
                                sender.sendMessage("§cТолько игроки!");
                                return;
                            }

                            Player player = Bukkit.getPlayer(
                                    sender.getPlatformPlayer().getUniqueId()
                            );

                            if (player == null) return;

                            MLMenuGUI.openMenu(player, 0);
                        })
        );

        // ========== /grimml reload - Перезагрузить конфигурацию ==========
        manager.command(
                manager.commandBuilder("grimml", "ml")
                        .literal("reload")
                        .permission("grim.ml.admin")
                        .handler(context -> {
                            Sender sender = context.sender();
                            sender.sendMessage("§e[GrimML] Перезагружаю конфигурацию...");

                            try {
                                if (config != null) {
                                    config.reload();
                                    sender.sendMessage("§a[GrimML] ✓ Конфигурация перезагружена!");
                                } else {
                                    sender.sendMessage("§c[GrimML] ✗ Config не инициализирован!");
                                }
                            } catch (Exception e) {
                                sender.sendMessage("§c[GrimML] Ошибка: " + e.getMessage());
                                e.printStackTrace();
                            }
                        })
        );

        // ========== /grimml balance - Проверить баланс датасетов ==========
        manager.command(
                manager.commandBuilder("grimml", "ml")
                        .literal("balance", "bal")
                        .permission("grim.ml.admin")
                        .handler(context -> {
                            Sender sender = context.sender();
                            sender.sendMessage("§e[GrimML] Анализирую датасеты...");

                            try {
                                DatasetBalanceChecker.BalanceReport report =
                                        DatasetBalanceChecker.analyzeBalance();

                                // Отправляем отчёт
                                String[] lines = report.getFormattedReport().split("\n");
                                for (String line : lines) {
                                    sender.sendMessage(line);
                                }

                                // Дополнительные советы
                                if (!report.canTrain()) {
                                    sender.sendMessage("");
                                    sender.sendMessage("§c⚠ ВНИМАНИЕ: Обучение может дать плохие результаты!");
                                    sender.sendMessage("§e   Соберите больше данных перед обучением");
                                } else if (report.isBalanced()) {
                                    sender.sendMessage("");
                                    sender.sendMessage("§a✓ Всё готово! Можно обучать модель:");
                                    sender.sendMessage("§a   /grimAiTrain");
                                }

                            } catch (Exception e) {
                                sender.sendMessage("§c[GrimML] Ошибка анализа: " + e.getMessage());
                                e.printStackTrace();
                            }
                        })
        );

        // ========== /grimml list - Открыть GUI ==========
        manager.command(
                manager.commandBuilder("grimml", "ml")
                        .literal("list", "gui")
                        .permission("grim.ml.list")
                        .handler(context -> {
                            Sender sender = context.sender();

                            if (!sender.isPlayer()) {
                                sender.sendMessage("§c[GrimML] Только игроки!");
                                return;
                            }

                            Player player = Bukkit.getPlayer(
                                    sender.getPlatformPlayer().getUniqueId()
                            );

                            if (player == null) {
                                sender.sendMessage("§c[GrimML] Игрок не найден!");
                                return;
                            }

                            MLMenuGUI.openMenu(player, 0);
                        })
        );

        // ========== /grimml vl <player> - Показать VL игрока ==========
        manager.command(
                manager.commandBuilder("grimml", "ml")
                        .literal("vl")
                        .required("player", StringParser.stringParser())
                        .permission("grim.ml.admin")
                        .handler(context -> {
                            Sender sender = context.sender();
                            String playerName = context.get("player");

                            Player target = Bukkit.getPlayer(playerName);
                            if (target == null) {
                                sender.sendMessage("§c[GrimML] Игрок не найден!");
                                return;
                            }

                            if (penaltySystem == null) {
                                sender.sendMessage("§c[GrimML] PenaltySystem не инициализирован!");
                                return;
                            }

                            int vl = penaltySystem.getVL(target.getUniqueId());
                            sender.sendMessage("§e[GrimML] VL игрока §a" + target.getName() + "§e: §c" + vl);
                        })
        );

        // ========== /grimml vl <player> reset - Сбросить VL ==========
        manager.command(
                manager.commandBuilder("grimml", "ml")
                        .literal("vl")
                        .required("player", StringParser.stringParser())
                        .literal("reset")
                        .permission("grim.ml.admin")
                        .handler(context -> {
                            Sender sender = context.sender();
                            String playerName = context.get("player");

                            Player target = Bukkit.getPlayer(playerName);
                            if (target == null) {
                                sender.sendMessage("§c[GrimML] Игрок не найден!");
                                return;
                            }

                            if (penaltySystem == null) {
                                sender.sendMessage("§c[GrimML] PenaltySystem не инициализирован!");
                                return;
                            }

                            int oldVL = penaltySystem.getVL(target.getUniqueId());
                            penaltySystem.resetVL(target.getUniqueId());

                            sender.sendMessage("§a[GrimML] VL сброшен: " + target.getName() +
                                    " (" + oldVL + " → 0)");
                        })
        );

        // ========== /grimml vl <player> set <amount> - Установить VL ==========
        manager.command(
                manager.commandBuilder("grimml", "ml")
                        .literal("vl")
                        .required("player", StringParser.stringParser())
                        .literal("set")
                        .required("amount", IntegerParser.integerParser(0, 100))
                        .permission("grim.ml.admin")
                        .handler(context -> {
                            Sender sender = context.sender();
                            String playerName = context.get("player");
                            int amount = context.get("amount");

                            Player target = Bukkit.getPlayer(playerName);
                            if (target == null) {
                                sender.sendMessage("§c[GrimML] Игрок не найден!");
                                return;
                            }

                            if (penaltySystem == null) {
                                sender.sendMessage("§c[GrimML] PenaltySystem не инициализирован!");
                                return;
                            }

                            int oldVL = penaltySystem.getVL(target.getUniqueId());

                            // Сбросить и установить новый
                            penaltySystem.resetVL(target.getUniqueId());
                            for (int i = 0; i < amount; i++) {
                                penaltySystem.incrementVL(target.getUniqueId());
                            }

                            sender.sendMessage("§a[GrimML] VL установлен: " + target.getName() +
                                    " (" + oldVL + " → " + amount + ")");
                        })
        );

        // ========== /grimml history - История наказаний ==========
        manager.command(
                manager.commandBuilder("grimml", "ml")
                        .literal("history", "h")
                        .permission("grim.ml.admin")
                        .handler(context -> {
                            Sender sender = context.sender();

                            if (penaltySystem == null) {
                                sender.sendMessage("§c[GrimML] PenaltySystem не инициализирован!");
                                return;
                            }

                            var history = penaltySystem.getHistory();

                            if (history.isEmpty()) {
                                sender.sendMessage("§e[GrimML] История наказаний пуста");
                                return;
                            }

                            sender.sendMessage("§6§l╔═══════════════════════════════════════════╗");
                            sender.sendMessage("§6§l║     ИСТОРИЯ НАКАЗАНИЙ ML                  ║");
                            sender.sendMessage("§6§l╚═══════════════════════════════════════════╝");
                            sender.sendMessage("");

                            int count = 0;
                            for (PenaltySystem.PenaltyRecord record : history) {
                                if (count >= 10) break; // Показываем последние 10

                                sender.sendMessage(String.format(
                                        "§7[%s] §c%s §7- §e%s §7(VL: §c%d§7, Prob: §e%.2f§7)",
                                        record.getFormattedTimestamp(),
                                        record.playerName,
                                        record.actionType,
                                        record.violationLevel,
                                        record.probability
                                ));

                                count++;
                            }

                            sender.sendMessage("");
                            sender.sendMessage("§7Показано: " + Math.min(10, history.size()) +
                                    " из " + history.size());
                        })
        );

        // ========== /grimml stats - Статистика системы ==========
        manager.command(
                manager.commandBuilder("grimml", "ml")
                        .literal("stats", "statistics")
                        .permission("grim.ml.admin")
                        .handler(context -> {
                            Sender sender = context.sender();

                            sender.sendMessage("§6§l╔═══════════════════════════════════════════╗");
                            sender.sendMessage("§6§l║     СТАТИСТИКА ML СИСТЕМЫ                 ║");
                            sender.sendMessage("§6§l╚═══════════════════════════════════════════╝");
                            sender.sendMessage("");

                            // Config stats
                            if (config != null) {
                                sender.sendMessage("§b⚙ КОНФИГУРАЦИЯ:");
                                sender.sendMessage("§7   AI Threshold: §e" +
                                        String.format("%.2f", config.getAIThreshold()));
                                sender.sendMessage("§7   Sequence Length: §e" +
                                        config.getAISequenceLength());
                                sender.sendMessage("§7   Step Interval: §e" +
                                        config.getAIStepInterval());
                                sender.sendMessage("");

                                // Hologram stats
                                sender.sendMessage("§b👁 ГОЛОГРАММЫ:");
                                sender.sendMessage("§7   Enabled: " +
                                        (config.isHologramEnabled() ? "§a✓" : "§c✗"));
                                sender.sendMessage("§7   OP Only: " +
                                        (config.isHologramOpOnly() ? "§a✓" : "§c✗"));
                                sender.sendMessage("§7   Max Strikes: §e" +
                                        config.getHologramMaxStrikes());
                                sender.sendMessage("");
                            }

                            // Penalty stats
                            if (penaltySystem != null) {
                                sender.sendMessage("§b⚖ НАКАЗАНИЯ:");
                                sender.sendMessage("§7   Min Probability: §e" +
                                        String.format("%.2f", penaltySystem.getMinProbability()));
                                sender.sendMessage("§7   VL Decay: " +
                                        (penaltySystem.isVLDecayEnabled() ? "§a✓" : "§c✗"));
                                sender.sendMessage("§7   Decay Amount: §e" +
                                        penaltySystem.getVLDecayAmount());
                                sender.sendMessage("§7   Decay Interval: §e" +
                                        penaltySystem.getVLDecayInterval() + "s");
                                sender.sendMessage("");

                                // History stats
                                int historySize = penaltySystem.getHistory().size();
                                sender.sendMessage("§b📜 ИСТОРИЯ:");
                                sender.sendMessage("§7   Total Penalties: §e" + historySize);
                            }

                            sender.sendMessage("");
                            sender.sendMessage("§8════════════════════════════════════════════");
                        })
        );

        // ========== /grimml help - Помощь ==========
        manager.command(
                manager.commandBuilder("grimml", "ml")
                        .literal("help", "?")
                        .permission("grim.ml.admin")
                        .handler(context -> {
                            Sender sender = context.sender();

                            sender.sendMessage("§6§l╔═══════════════════════════════════════════╗");
                            sender.sendMessage("§6§l║          GRIMML КОМАНДЫ v2.0              ║");
                            sender.sendMessage("§6§l╚═══════════════════════════════════════════╝");
                            sender.sendMessage("");
                            sender.sendMessage("§e/grimml reload §7- перезагрузить конфигурацию");
                            sender.sendMessage("§e/grimml balance §7- проверить баланс датасетов");
                            sender.sendMessage("§e/grimml list §7- открыть GUI меню");
                            sender.sendMessage("§e/grimml vl <player> §7- показать VL игрока");
                            sender.sendMessage("§e/grimml vl <player> reset §7- сбросить VL");
                            sender.sendMessage("§e/grimml vl <player> set <amount> §7- установить VL");
                            sender.sendMessage("§e/grimml history §7- история наказаний");
                            sender.sendMessage("§e/grimml stats §7- статистика системы");
                            sender.sendMessage("");
                            sender.sendMessage("§7Алиасы: §e/ml §7= §e/grimml");
                            sender.sendMessage("§8════════════════════════════════════════════");
                        })
        );
    }
}
