package ac.grim.grimac.command.commands;

import ac.grim.grimac.GrimAPI;
import ac.grim.grimac.checks.impl.aim.ImprovedAimML;
import ac.grim.grimac.command.BuildableCommand;
import ac.grim.grimac.platform.api.command.PlayerSelector;
import ac.grim.grimac.platform.api.manager.cloud.CloudCommandAdapter;
import ac.grim.grimac.platform.api.player.PlatformPlayer;
import ac.grim.grimac.platform.api.sender.Sender;
import ac.grim.grimac.player.GrimPlayer;
import ac.grim.grimac.utils.anticheat.MessageUtil;
import org.incendo.cloud.CommandManager;
import org.incendo.cloud.context.CommandContext;
import org.incendo.cloud.parser.standard.StringParser;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.Objects;

/**
 * Улучшенные команды для ML-системы (по образцу SlothAC)
 *
 * Локальный сбор:
 * /grimAiOn - Начать запись для себя
 * /grimAiOff - Остановить и сохранить
 * /grimAiDiscard - Отменить запись
 *
 * Глобальный сбор:
 * /grimAiGlobal start <id> - Начать сбор со всех игроков
 * /grimAiGlobal stop - Остановить и создать ZIP-архив
 *
 * Управление:
 * /grimAiTrain - Обучить модель
 * /grimAiStatus [игрок] - Показать статус
 * /grimAiList - Список датасетов
 * /shame ban <игрок> - Забанить читера
 */
public class ImprovedMLCommands implements BuildableCommand {

    @Override
    public void register(CommandManager<Sender> commandManager, CloudCommandAdapter adapter) {

        // ========== /grimAiOn - Локальная запись ==========
        commandManager.command(
                commandManager.commandBuilder("grimAiOn")
                        .permission("grim.ml.record")
                        .handler(this::handleRecordStart)
        );

        // ========== /grimAiOff - Остановить запись ==========
        commandManager.command(
                commandManager.commandBuilder("grimAiOff")
                        .permission("grim.ml.record")
                        .handler(this::handleRecordStop)
        );

        // ========== /grimAiDiscard - Отменить запись ==========
        commandManager.command(
                commandManager.commandBuilder("grimAiDiscard")
                        .permission("grim.ml.record")
                        .handler(this::handleRecordDiscard)
        );

        // ========== /grimAiGlobal start <id> - Глобальный сбор ==========
        commandManager.command(
                commandManager.commandBuilder("grimAiGlobal")
                        .literal("start")
                        .permission("grim.ml.global")
                        .required("id", StringParser.stringParser())
                        .handler(this::handleGlobalStart)
        );

        // ========== /grimAiGlobal stop - Остановить глобальный сбор ==========
        commandManager.command(
                commandManager.commandBuilder("grimAiGlobal")
                        .literal("stop")
                        .permission("grim.ml.global")
                        .handler(this::handleGlobalStop)
        );

        // ========== /grimAiTrain - Обучить модель ==========
        commandManager.command(
                commandManager.commandBuilder("grimAiTrain")
                        .permission("grim.ml.train")
                        .handler(this::handleTrain)
        );

        // ========== /grimAiStatus [игрок] - Статус ==========
        commandManager.command(
                commandManager.commandBuilder("grimAiStatus")
                        .permission("grim.ml.status")
                        .optional("target", adapter.singlePlayerSelectorParser())
                        .handler(this::handleStatus)
        );

        // ========== /grimAiList - Список датасетов ==========
        commandManager.command(
                commandManager.commandBuilder("grimAiList")
                        .permission("grim.ml.list")
                        .handler(this::handleList)
        );

        // ========== /shame ban <игрок> - Забанить читера ==========
        commandManager.command(
                commandManager.commandBuilder("shame")
                        .literal("ban")
                        .permission("grim.ml.shame")
                        .required("target", adapter.singlePlayerSelectorParser())
                        .handler(this::handleShameBan)
        );
    }

    // ========== ОБРАБОТЧИКИ КОМАНД ==========

    private void handleRecordStart(@NotNull CommandContext<Sender> context) {
        Sender sender = context.sender();

        if (!sender.isPlayer()) {
            sender.sendMessage("§c[GrimAC ML] Только игроки могут использовать эту команду!");
            return;
        }

        PlatformPlayer platformPlayer = sender.getPlatformPlayer();
        if (platformPlayer == null) {
            sender.sendMessage("§c[GrimAC ML] Ошибка получения данных игрока!");
            return;
        }

        GrimPlayer grimPlayer = GrimAPI.INSTANCE.getPlayerDataManager().getPlayer(platformPlayer.getUniqueId());
        if (grimPlayer == null) {
            sender.sendMessage("§c[GrimAC ML] Игрок не найден!");
            return;
        }

        if (ImprovedAimML.getActiveCollectors().containsKey(grimPlayer.uuid)) {
            sender.sendMessage("§c[GrimAC ML] Запись уже активна! Используйте /grimAiOff");
            return;
        }

        // Создаём уникальный ID для этой сессии
        String sessionId = "LOCAL_" + grimPlayer.getName() + "_" + System.currentTimeMillis();
        ImprovedAimML.startRecording(grimPlayer, sessionId);

        sender.sendMessage("§a[GrimAC ML] ═══════════════════════════════");
        sender.sendMessage("§a§l✓ Запись датасета начата!");
        sender.sendMessage("§7");
        sender.sendMessage("§e⚠ ВАЖНО:");
        sender.sendMessage("§7• §fАктивно двигайте мышкой во время боя!");
        sender.sendMessage("§7• §fМинимум §e100 тиков §7для сохранения");
        sender.sendMessage("§7• §fРекомендуется: §a500+ тиков");
        sender.sendMessage("§7");
        sender.sendMessage("§7Проверка: §b/grimAiStatus");
        sender.sendMessage("§7Легит → §a/grimAiOff");
        sender.sendMessage("§7Читы → §c/shame ban §7<ник>");
        sender.sendMessage("§a[GrimAC ML] ═══════════════════════════════");
    }

    private void handleRecordStop(@NotNull CommandContext<Sender> context) {
        Sender sender = context.sender();

        if (!sender.isPlayer()) {
            sender.sendMessage("§c[GrimAC ML] Только игроки могут использовать эту команду!");
            return;
        }

        PlatformPlayer platformPlayer = sender.getPlatformPlayer();
        if (platformPlayer == null) {
            sender.sendMessage("§c[GrimAC ML] Ошибка получения данных игрока!");
            return;
        }

        GrimPlayer grimPlayer = GrimAPI.INSTANCE.getPlayerDataManager().getPlayer(platformPlayer.getUniqueId());
        if (grimPlayer == null) {
            sender.sendMessage("§c[GrimAC ML] Игрок не найден!");
            return;
        }

        if (!ImprovedAimML.getActiveCollectors().containsKey(grimPlayer.uuid)) {
            sender.sendMessage("§c[GrimAC ML] Запись не активна! Используйте /grimAiOn");
            return;
        }

        // Проверяем количество тиков ДО сохранения
        int tickCount = ImprovedAimML.getTickCount(grimPlayer.uuid);
        if (tickCount < 100) {
            sender.sendMessage("§c[GrimAC ML] ═══════════════════════════════");
            sender.sendMessage("§c§l✗ Недостаточно данных!");
            sender.sendMessage("§7");
            sender.sendMessage("§7Собрано: §c" + tickCount + " §7тиков");
            sender.sendMessage("§7Минимум: §e100 §7тиков");
            sender.sendMessage("§7Рекомендуется: §a500+ §7тиков");
            sender.sendMessage("§7");
            sender.sendMessage("§e⚠ Совет: §fАктивно двигайте мышкой во время боя!");
            sender.sendMessage("§7Используйте §c/grimAiDiscard §7для отмены");
            sender.sendMessage("§c[GrimAC ML] ═══════════════════════════════");
            return;
        }

        String result = ImprovedAimML.stopRecording(grimPlayer, true);
        sender.sendMessage("§a[GrimAC ML] ═══════════════════════════════");
        sender.sendMessage("§a§l✓ " + result);
        sender.sendMessage("§a[GrimAC ML] ═══════════════════════════════");
    }

    private void handleRecordDiscard(@NotNull CommandContext<Sender> context) {
        Sender sender = context.sender();

        if (!sender.isPlayer()) {
            sender.sendMessage("§c[GrimAC ML] Только игроки могут использовать эту команду!");
            return;
        }

        PlatformPlayer platformPlayer = sender.getPlatformPlayer();
        if (platformPlayer == null) {
            sender.sendMessage("§c[GrimAC ML] Ошибка получения данных игрока!");
            return;
        }

        GrimPlayer grimPlayer = GrimAPI.INSTANCE.getPlayerDataManager().getPlayer(platformPlayer.getUniqueId());
        if (grimPlayer == null) {
            sender.sendMessage("§c[GrimAC ML] Игрок не найден!");
            return;
        }

        String result = ImprovedAimML.stopRecording(grimPlayer, false);
        sender.sendMessage("§e[GrimAC ML] " + result);
    }

    private void handleGlobalStart(@NotNull CommandContext<Sender> context) {
        Sender sender = context.sender();
        String collectionId = context.get("id");

        if (ImprovedAimML.getGlobalCollectionId() != null) {
            sender.sendMessage("§c[GrimAC ML] Глобальный сбор уже активен!");
            sender.sendMessage("§7ID: " + ImprovedAimML.getGlobalCollectionId());
            sender.sendMessage("§7Остановите его: §e/grimAiGlobal stop");
            return;
        }

        int started = ImprovedAimML.startGlobalCollection(collectionId);

        sender.sendMessage("§a[GrimAC ML] ═══════════════════════════════");
        sender.sendMessage("§a§l✓ Глобальный сбор начат!");
        sender.sendMessage("§7");
        sender.sendMessage("§7ID сессии: §e" + collectionId);
        sender.sendMessage("§7Игроков: §e" + started);
        sender.sendMessage("§7");
        sender.sendMessage("§7Остановить: §c/grimAiGlobal stop");
        sender.sendMessage("§a[GrimAC ML] ═══════════════════════════════");

        // Оповещаем всех админов
        for (GrimPlayer online : GrimAPI.INSTANCE.getPlayerDataManager().getEntries()) {
            if (online.hasPermission("grim.alerts")) {
                online.sendMessage(String.format(
                    "§e[GrimAC ML] Начат глобальный сбор датасетов: §b%s §7(%d игроков)",
                    collectionId,
                    started
                ));
            }
        }
    }

    private void handleGlobalStop(@NotNull CommandContext<Sender> context) {
        Sender sender = context.sender();

        if (ImprovedAimML.getGlobalCollectionId() == null) {
            sender.sendMessage("§c[GrimAC ML] Глобальный сбор не активен!");
            return;
        }

        String result = ImprovedAimML.stopGlobalCollection();

        sender.sendMessage("§a[GrimAC ML] ═══════════════════════════════");
        sender.sendMessage("§a§l✓ " + result);
        sender.sendMessage("§7");
        sender.sendMessage("§7Архив будет создан в: §e/plugins/GrimAC/ml_datasets/");
        sender.sendMessage("§a[GrimAC ML] ═══════════════════════════════");

        // Оповещаем всех админов
        for (GrimPlayer online : GrimAPI.INSTANCE.getPlayerDataManager().getEntries()) {
            if (online.hasPermission("grim.alerts")) {
                online.sendMessage("§a[GrimAC ML] Глобальный сбор датасетов завершён!");
            }
        }
    }

    private void handleTrain(@NotNull CommandContext<Sender> context) {
        Sender sender = context.sender();

        sender.sendMessage("§e[GrimAC ML] Начинаем обучение модели...");
        sender.sendMessage("§7Это может занять некоторое время...");

        // Запускаем в отдельном потоке
        GrimAPI.INSTANCE.getScheduler().getAsyncScheduler().runNow(
                GrimAPI.INSTANCE.getGrimPlugin(),
                () -> {
                    String result = ImprovedAimML.trainModel();
                    sender.sendMessage(result);
                }
        );
    }

    private void handleStatus(@NotNull CommandContext<Sender> context) {
        Sender sender = context.sender();

        GrimPlayer targetPlayer = null;

        if (context.contains("target")) {
            PlayerSelector selector = context.get("target");
            PlatformPlayer platformPlayer = selector.getSinglePlayer().getPlatformPlayer();

            if (platformPlayer != null && !platformPlayer.isExternalPlayer()) {
                targetPlayer = GrimAPI.INSTANCE.getPlayerDataManager().getPlayer(platformPlayer.getUniqueId());
            }
        } else if (sender instanceof PlatformPlayer platformPlayer) {
            targetPlayer = GrimAPI.INSTANCE.getPlayerDataManager().getPlayer(platformPlayer.getUniqueId());
        }

        if (targetPlayer == null) {
            sender.sendMessage("§c[GrimAC ML] Игрок не найден!");
            return;
        }

        String status = ImprovedAimML.getStatus(targetPlayer);
        sender.sendMessage(status);

        // Предупреждение если мало данных
        int tickCount = ImprovedAimML.getTickCount(targetPlayer.uuid);
        if (ImprovedAimML.getActiveCollectors().containsKey(targetPlayer.uuid) && tickCount < 100) {
            sender.sendMessage("§e⚠ ВНИМАНИЕ: Мало данных! Активнее двигайте мышкой!");
        }
    }

    private void handleList(@NotNull CommandContext<Sender> context) {
        Sender sender = context.sender();

        List<String> datasets = ImprovedAimML.listDatasets();

        if (datasets.isEmpty()) {
            sender.sendMessage("§c[GrimAC ML] Датасеты не найдены!");
            sender.sendMessage("§7Используйте /grimAiOn или /grimAiGlobal start");
            return;
        }

        sender.sendMessage("§b§l[GrimAC ML Датасеты]");
        sender.sendMessage("§7Всего файлов: §e" + datasets.size());
        sender.sendMessage("");

        int cheatCount = 0, legitCount = 0, zipCount = 0;

        for (String dataset : datasets) {
            if (dataset.endsWith(".zip")) {
                zipCount++;
                sender.sendMessage("§d[АРХИВ] §7" + dataset);
            } else if (dataset.startsWith("CHEAT_")) {
                cheatCount++;
                sender.sendMessage("§c[ЧИТ] §7" + dataset);
            } else if (dataset.startsWith("LEGIT_")) {
                legitCount++;
                sender.sendMessage("§a[ЛЕГИТ] §7" + dataset);
            } else {
                sender.sendMessage("§e[?] §7" + dataset);
            }
        }

        sender.sendMessage("");
        sender.sendMessage(String.format(
            "§7Читы: §c%d §7| Легит: §a%d §7| Архивы: §d%d",
            cheatCount, legitCount, zipCount
        ));
    }

    private void handleShameBan(@NotNull CommandContext<Sender> context) {
        Sender sender = context.sender();
        PlayerSelector target = context.get("target");

        PlatformPlayer targetPlatformPlayer = target.getSinglePlayer().getPlatformPlayer();
        if (Objects.requireNonNull(targetPlatformPlayer).isExternalPlayer()) {
            sender.sendMessage("§c[GrimAC ML] Игрок не на этом сервере!");
            return;
        }

        GrimPlayer grimPlayer = GrimAPI.INSTANCE.getPlayerDataManager().getPlayer(targetPlatformPlayer.getUniqueId());
        if (grimPlayer == null) {
            sender.sendMessage("§c[GrimAC ML] Игрок не найден!");
            return;
        }

        // Проверяем наличие активной записи
        if (!ImprovedAimML.getActiveCollectors().containsKey(grimPlayer.uuid)) {
            sender.sendMessage("§c[GrimAC ML] ═══════════════════════════════");
            sender.sendMessage("§c§l✗ У игрока нет активной записи!");
            sender.sendMessage("§7");
            sender.sendMessage("§7Сначала: §e/grimAiOn");
            sender.sendMessage("§7Затем играйте с читами 5-10 минут");
            sender.sendMessage("§7После: §c/shame ban " + grimPlayer.getName());
            sender.sendMessage("§c[GrimAC ML] ═══════════════════════════════");
            return;
        }

        // Проверяем количество тиков
        int tickCount = ImprovedAimML.getTickCount(grimPlayer.uuid);
        if (tickCount < 100) {
            sender.sendMessage("§c[GrimAC ML] ═══════════════════════════════");
            sender.sendMessage("§c§l✗ Недостаточно данных для бана!");
            sender.sendMessage("§7");
            sender.sendMessage("§7Собрано: §c" + tickCount + " §7тиков");
            sender.sendMessage("§7Минимум: §e100 §7тиков");
            sender.sendMessage("§7Рекомендуется: §a500+ §7тиков");
            sender.sendMessage("§7");
            sender.sendMessage("§7Используйте §b/grimAiStatus " + grimPlayer.getName());
            sender.sendMessage("§c[GrimAC ML] ═══════════════════════════════");
            return;
        }

        // Помечаем как читера
        if (!ImprovedAimML.markAsCheater(grimPlayer.uuid)) {
            sender.sendMessage("§c[GrimAC ML] Ошибка маркировки датасета!");
            return;
        }

        // Сохраняем
        String saveResult = ImprovedAimML.stopRecording(grimPlayer, true);

        // Баним
        String banReason = "§c[GrimAC ML] Обнаружено использование читов (ML Detection)";
        grimPlayer.disconnect(MessageUtil.miniMessage(banReason));

        // Оповещаем
        sender.sendMessage("§c[GrimAC ML] ═══════════════════════════════");
        sender.sendMessage("§c§l✓ Игрок " + grimPlayer.getName() + " забанен!");
        sender.sendMessage("§7" + saveResult);
        sender.sendMessage("§c[GrimAC ML] ═══════════════════════════════");

        // Алерты для всех админов
        for (GrimPlayer online : GrimAPI.INSTANCE.getPlayerDataManager().getEntries()) {
            if (online.hasPermission("grim.alerts")) {
                online.sendMessage(String.format(
                        "§c[SHAME BAN] §f%s §7забанен за читы игроком §f%s §7(Тиков: §e%d§7)",
                        grimPlayer.getName(),
                        sender.getName(),
                        tickCount
                ));
            }
        }
    }
}
