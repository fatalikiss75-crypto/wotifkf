package ac.grim.grimac.platform.bukkit.command;

import ac.grim.grimac.checks.impl.aim.DatasetBalanceChecker;
import ac.grim.grimac.command.BuildableCommand;
import ac.grim.grimac.platform.api.manager.cloud.CloudCommandAdapter;
import ac.grim.grimac.platform.api.sender.Sender;
import org.incendo.cloud.CommandManager;
import org.incendo.cloud.context.CommandContext;
import org.jetbrains.annotations.NotNull;

/**
 * Команда /grimAiBalance - проверить баланс датасетов
 */
public class MLBalanceCommand implements BuildableCommand {

    @Override
    public void register(CommandManager<Sender> commandManager, CloudCommandAdapter adapter) {

        // /grimAiBalance
        commandManager.command(
                commandManager.commandBuilder("grimAiBalance")
                        .permission("grim.ml.admin")
                        .handler(this::handleBalance)
        );
    }

    private void handleBalance(@NotNull CommandContext<Sender> context) {
        Sender sender = context.sender();

        sender.sendMessage("§e[GrimAC ML] Анализирую датасеты...");

        try {
            DatasetBalanceChecker.BalanceReport report = DatasetBalanceChecker.analyzeBalance();

            // Отправляем отчёт
            String[] lines = report.getFormattedReport().split("\n");
            for (String line : lines) {
                sender.sendMessage(line);
            }

            // Дополнительные советы
            if (!report.canTrain()) {
                sender.sendMessage("");
                sender.sendMessage("§c⚠ ВНИМАНИЕ: Обучение модели может дать плохие результаты!");
                sender.sendMessage("§e   Соберите больше данных перед обучением");
            } else if (report.isBalanced()) {
                sender.sendMessage("");
                sender.sendMessage("§a✓ Всё готово! Можно обучать модель:");
                sender.sendMessage("§a   /grimAiTrain");
            }

        } catch (Exception e) {
            sender.sendMessage("§c[GrimAC ML] Ошибка анализа: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
