package ac.grim.grimac.platform.bukkit.player;

import ac.grim.grimac.checks.impl.aim.MLBridgeHolder;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.inventory.InventoryClickEvent;
import org.bukkit.event.inventory.InventoryCloseEvent;
import org.bukkit.event.player.PlayerQuitEvent;

/**
 * Обработчик событий для ML GUI меню
 * FIXED: Использует getTitle() вместо title()
 */
public class MLMenuListener implements Listener {

    @EventHandler
    public void onInventoryClick(InventoryClickEvent event) {
        if (!(event.getWhoClicked() instanceof Player player)) {
            return;
        }

        // ФИКС: Используем getTitle() вместо title()
        String title = event.getView().getTitle();

        if (!title.contains("Курятник")) {
            return;
        }

        event.setCancelled(true);

        int slot = event.getRawSlot();

        if (slot < 0 || slot >= 54) {
            return;
        }

        MLMenuGUI.handleClick(player, slot);
    }

    @EventHandler
    public void onInventoryClose(InventoryCloseEvent event) {
        if (!(event.getPlayer() instanceof Player player)) {
            return;
        }

        // ФИКС: Используем getTitle() вместо title()
        String title = event.getView().getTitle();

        if (title.contains("Курятник")) {
            MLMenuGUI.removeViewer(player.getUniqueId());
        }
    }

    @EventHandler
    public void onPlayerQuit(PlayerQuitEvent event) {
        Player player = event.getPlayer();

        // Очищаем данные просмотра меню
        MLMenuGUI.removeViewer(player.getUniqueId());

        // Удаляем голограмму ЧЕРЕЗ BRIDGE
        MLBridgeHolder.getBridge().removeHologram(player.getUniqueId());
    }
}
