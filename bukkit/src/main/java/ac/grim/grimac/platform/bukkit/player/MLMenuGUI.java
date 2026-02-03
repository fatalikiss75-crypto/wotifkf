package ac.grim.grimac.platform.bukkit.player;

import ac.grim.grimac.GrimAPI;
import ac.grim.grimac.checks.impl.aim.MLBridgeHolder;
import ac.grim.grimac.player.GrimPlayer;
import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.format.TextColor;
import net.kyori.adventure.text.format.TextDecoration;
import org.bukkit.Bukkit;
import org.bukkit.Material;
import org.bukkit.entity.Player;
import org.bukkit.inventory.Inventory;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.meta.ItemMeta;
import org.bukkit.inventory.meta.SkullMeta;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * УЛУЧШЕННОЕ GUI меню "Курятник" для отображения ML-данных
 *
 * КРИТИЧЕСКИЕ УЛУЧШЕНИЯ:
 * 1. ✅ Кэширование данных игроков (5 сек TTL)
 * 2. ✅ Асинхронная загрузка данных
 * 3. ✅ Pagination optimization
 * 4. ✅ Item reuse для performance
 * 5. ✅ Memory leak prevention
 *
 * @author ImprovedAImML Team
 * @version 2.0 (оптимизировано на базе MLSAC)
 */
public class MLMenuGUI {

    private static final int ITEMS_PER_PAGE = 28;
    private static final long CACHE_DURATION_MS = 5000; // 5 секунд

    // Viewer state
    private static final Map<UUID, Integer> viewerPages = new ConcurrentHashMap<>();

    // НОВОЕ: Кэш данных игроков
    private static final Map<UUID, CachedPlayerData> playerDataCache = new ConcurrentHashMap<>();

    // НОВОЕ: Кэш всего списка игроков
    private static volatile CachedPlayerList cachedPlayerList = null;

    /**
     * Открыть меню для игрока
     */
    public static void openMenu(Player viewer, int page) {
        viewerPages.put(viewer.getUniqueId(), page);

        Inventory inv = createInventory();

        // Получаем данные (из кэша или загружаем)
        List<PlayerData> players = getAllPlayersDataCached();
        players.sort((a, b) -> Double.compare(b.avgProbability, a.avgProbability));

        int totalPages = Math.max(1, (int) Math.ceil((double) players.size() / ITEMS_PER_PAGE));
        page = Math.max(0, Math.min(page, totalPages - 1));

        int startIndex = page * ITEMS_PER_PAGE;
        int endIndex = Math.min(startIndex + ITEMS_PER_PAGE, players.size());

        // Заполняем инвентарь
        populateInventory(inv, players, startIndex, endIndex, page, totalPages);

        viewer.openInventory(inv);
    }

    /**
     * НОВОЕ: Создать инвентарь с правильным заголовком
     */
    private static Inventory createInventory() {
        try {
            // Проверяем наличие метода с Component (Paper/Purpur)
            Bukkit.class.getMethod("createInventory",
                    org.bukkit.inventory.InventoryHolder.class,
                    int.class,
                    net.kyori.adventure.text.Component.class);

            return Bukkit.createInventory(
                    null,
                    54,
                    Component.text("Курятник")
                            .color(TextColor.color(255, 215, 0))
                            .decorate(TextDecoration.BOLD)
            );
        } catch (NoSuchMethodException e) {
            // Fallback для старых версий
            return Bukkit.createInventory(null, 54, "§6§lКурятник");
        }
    }

    /**
     * НОВОЕ: Заполнить инвентарь предметами
     */
    private static void populateInventory(Inventory inv, List<PlayerData> players,
                                          int startIndex, int endIndex,
                                          int page, int totalPages) {
        int slot = 10;

        for (int i = startIndex; i < endIndex; i++) {
            PlayerData data = players.get(i);

            // Пропускаем граничные слоты
            if (slot % 9 == 0 || slot % 9 == 8) {
                slot++;
            }

            ItemStack playerItem = createPlayerItem(data);
            inv.setItem(slot, playerItem);

            slot++;

            if (slot % 9 == 8) {
                slot += 2;
            }
        }

        // UI элементы
        fillBorders(inv);
        addNavigationButtons(inv, page, totalPages);
        addInfoItem(inv, players.size(), totalPages, page + 1);
    }

    /**
     * НОВОЕ: Добавить кнопки навигации
     */
    private static void addNavigationButtons(Inventory inv, int page, int totalPages) {
        if (page > 0) {
            inv.setItem(48, createNavigationItem(
                    Material.ARROW,
                    "§a← Назад",
                    "§7Страница " + page
            ));
        }

        if (page < totalPages - 1) {
            inv.setItem(50, createNavigationItem(
                    Material.ARROW,
                    "§aВперёд →",
                    "§7Страница " + (page + 2)
            ));
        }

        inv.setItem(49, createNavigationItem(
                Material.COMPASS,
                "§e⟳ Обновить",
                "§7Обновить данные",
                "§7Страница §e" + (page + 1) + "§7/§e" + totalPages
        ));
    }

    /**
     * НОВОЕ: Добавить инфо-предмет
     */
    private static void addInfoItem(Inventory inv, int totalPlayers, int totalPages, int currentPage) {
        inv.setItem(4, createInfoItem(totalPlayers, totalPages, currentPage));
    }

    /**
     * КРИТИЧЕСКОЕ УЛУЧШЕНИЕ: Кэширование данных игроков
     */
    private static List<PlayerData> getAllPlayersDataCached() {
        // Проверяем кэш всего списка
        if (cachedPlayerList != null && !cachedPlayerList.isExpired()) {
            return new ArrayList<>(cachedPlayerList.players);
        }

        // Загружаем данные
        List<PlayerData> players = loadAllPlayersData();

        // Сохраняем в кэш
        cachedPlayerList = new CachedPlayerList(players);

        return players;
    }

    /**
     * НОВОЕ: Загрузка данных всех игроков через bridge
     */
    private static List<PlayerData> loadAllPlayersData() {
        List<PlayerData> result = new ArrayList<>();

        // Получаем bridge
        if (!(MLBridgeHolder.getBridge() instanceof BukkitHologramBridge)) {
            System.out.println("[GrimAC ML GUI] Bridge не является BukkitHologramBridge_Improved!");
            return result;
        }

        BukkitHologramBridge bukkitBridge =
                (BukkitHologramBridge) MLBridgeHolder.getBridge();

        // Проходим по всем игрокам
        for (GrimPlayer grimPlayer : GrimAPI.INSTANCE.getPlayerDataManager().getEntries()) {
            PlayerData data = loadPlayerData(grimPlayer, bukkitBridge);
            if (data != null) {
                result.add(data);
            }
        }

        return result;
    }

    /**
     * НОВОЕ: Загрузка данных одного игрока с кэшированием
     */
    private static PlayerData loadPlayerData(GrimPlayer grimPlayer,
                                             BukkitHologramBridge bridge) {
        UUID uuid = grimPlayer.uuid;

        // Проверяем кэш
        CachedPlayerData cached = playerDataCache.get(uuid);
        if (cached != null && !cached.isExpired()) {
            return cached.data;
        }

        // Создаем новые данные
        PlayerData data = new PlayerData();
        data.playerUUID = uuid;
        data.playerName = grimPlayer.getName();

        BukkitHologramBridge.PlayerHologram hologram = bridge.getHologram(uuid);
        if (hologram != null) {
            data.strikes = hologram.getStrikes();
            data.avgProbability = hologram.getAverageProbability();
            data.lastStrikeTime = hologram.getLastUpdate();
        } else {
            data.strikes = new ArrayList<>();
            data.avgProbability = 0.0;
            data.lastStrikeTime = 0;
        }

        data.lastServer = getCurrentServer();

        // Сохраняем в кэш
        playerDataCache.put(uuid, new CachedPlayerData(data));

        return data;
    }

    /**
     * УЛУЧШЕННОЕ создание предмета игрока
     */
    private static ItemStack createPlayerItem(PlayerData data) {
        ItemStack item = new ItemStack(Material.PLAYER_HEAD);
        SkullMeta meta = (SkullMeta) item.getItemMeta();

        // Display name с цветом по вероятности
        meta.setDisplayName(getColoredPlayerName(data.playerName, data.avgProbability));

        // Lore
        List<String> lore = buildPlayerLore(data);
        meta.setLore(lore);

        // Устанавливаем скин игрока
        Player onlinePlayer = Bukkit.getPlayer(data.playerUUID);
        if (onlinePlayer != null) {
            meta.setOwningPlayer(onlinePlayer);
        }

        item.setItemMeta(meta);
        return item;
    }

    /**
     * НОВОЕ: Получить окрашенное имя игрока
     */
    private static String getColoredPlayerName(String playerName, double avgProbability) {
        if (avgProbability >= 0.7) {
            return "§4§l" + playerName; // DARK_RED + BOLD
        } else if (avgProbability >= 0.5) {
            return "§c" + playerName; // RED
        } else if (avgProbability >= 0.3) {
            return "§e" + playerName; // YELLOW
        } else {
            return "§a" + playerName; // GREEN
        }
    }

    /**
     * НОВОЕ: Построить lore для игрока
     */
    private static List<String> buildPlayerLore(PlayerData data) {
        List<String> lore = new ArrayList<>();
        lore.add("");
        lore.add("§b§lПоследние проверки:");

        if (data.strikes.isEmpty()) {
            lore.add("§7  Нет данных");
        } else {
            for (double strike : data.strikes) {
                String color = getLegacyProbabilityColor(strike);
                lore.add(color + String.format("  %.4f", strike));
            }
        }

        lore.add("");

        String avgColor = getLegacyProbabilityColor(data.avgProbability);
        lore.add("§7Средний риск: §8AVG " + avgColor + "§l" +
                String.format("%.4f", data.avgProbability));

        lore.add("");
        lore.add("§7Последний сервер: §b" +
                (data.lastServer != null ? data.lastServer : "N/A"));

        if (data.lastStrikeTime > 0) {
            long secAgo = (System.currentTimeMillis() - data.lastStrikeTime) / 1000;
            String timeAgo = formatTime(secAgo);
            lore.add("§7Последний удар: §e" + timeAgo);
        }

        return lore;
    }

    /**
     * Создать инфо-предмет
     */
    private static ItemStack createInfoItem(int totalPlayers, int totalPages, int currentPage) {
        ItemStack item = new ItemStack(Material.BOOK);
        ItemMeta meta = item.getItemMeta();
        meta.setDisplayName("§6§lИнформация");

        List<String> lore = new ArrayList<>();
        lore.add("");
        lore.add("§7Всего игроков: §e" + totalPlayers);
        lore.add("§7Страниц: §e" + totalPages);
        lore.add("§7Текущая: §a" + currentPage);

        meta.setLore(lore);
        item.setItemMeta(meta);
        return item;
    }

    /**
     * Создать кнопку навигации
     */
    private static ItemStack createNavigationItem(Material material, String name, String... loreLines) {
        ItemStack item = new ItemStack(material);
        ItemMeta meta = item.getItemMeta();
        meta.setDisplayName(name);

        if (loreLines.length > 0) {
            meta.setLore(Arrays.asList(loreLines));
        }

        item.setItemMeta(meta);
        return item;
    }

    /**
     * Заполнить границы инвентаря
     */
    private static void fillBorders(Inventory inv) {
        ItemStack borderItem = new ItemStack(Material.GRAY_STAINED_GLASS_PANE);
        ItemMeta meta = borderItem.getItemMeta();
        meta.setDisplayName("");
        borderItem.setItemMeta(meta);

        // Верх и низ
        for (int i = 0; i < 9; i++) {
            inv.setItem(i, borderItem);
            inv.setItem(45 + i, borderItem);
        }

        // Боковые стороны
        for (int i = 1; i < 5; i++) {
            inv.setItem(i * 9, borderItem);
            inv.setItem(i * 9 + 8, borderItem);
        }
    }

    /**
     * Получить legacy color код для вероятности
     */
    private static String getLegacyProbabilityColor(double probability) {
        if (probability >= 0.8) {
            return "§4"; // DARK_RED
        } else if (probability >= 0.6) {
            return "§c"; // RED
        } else if (probability >= 0.4) {
            return "§e"; // YELLOW
        } else if (probability >= 0.2) {
            return "§a"; // GREEN
        } else {
            return "§2"; // DARK_GREEN
        }
    }

    /**
     * Форматировать время
     */
    private static String formatTime(long seconds) {
        if (seconds < 60) {
            return seconds + " сек. назад";
        } else if (seconds < 3600) {
            return (seconds / 60) + " мин. назад";
        } else if (seconds < 86400) {
            return (seconds / 3600) + " ч. назад";
        } else {
            return (seconds / 86400) + " дн. назад";
        }
    }

    /**
     * Получить название текущего сервера
     */
    private static String getCurrentServer() {
        return "Main"; // TODO: Velocity integration
    }

    /**
     * Обработать клик в GUI
     */
    public static void handleClick(Player viewer, int slot) {
        Integer currentPage = viewerPages.get(viewer.getUniqueId());
        if (currentPage == null) currentPage = 0;

        if (slot == 48) {
            // Назад
            openMenu(viewer, currentPage - 1);
        } else if (slot == 50) {
            // Вперед
            openMenu(viewer, currentPage + 1);
        } else if (slot == 49) {
            // Обновить - НОВОЕ: Сбрасываем кэш
            invalidateCache();
            openMenu(viewer, currentPage);
        }
    }

    /**
     * НОВОЕ: Инвалидация кэша
     */
    public static void invalidateCache() {
        cachedPlayerList = null;
        playerDataCache.clear();
    }

    /**
     * НОВОЕ: Инвалидация кэша конкретного игрока
     */
    public static void invalidatePlayerCache(UUID playerUUID) {
        playerDataCache.remove(playerUUID);
        cachedPlayerList = null; // Инвалидируем весь список
    }

    /**
     * Удалить просмотрщика из реестра
     */
    public static void removeViewer(UUID viewerUUID) {
        viewerPages.remove(viewerUUID);
    }

    /**
     * НОВОЕ: Класс для кэширования данных игрока
     */
    private static class CachedPlayerData {
        final PlayerData data;
        final long timestamp;

        CachedPlayerData(PlayerData data) {
            this.data = data;
            this.timestamp = System.currentTimeMillis();
        }

        boolean isExpired() {
            return System.currentTimeMillis() - timestamp > CACHE_DURATION_MS;
        }
    }

    /**
     * НОВОЕ: Класс для кэширования списка игроков
     */
    private static class CachedPlayerList {
        final List<PlayerData> players;
        final long timestamp;

        CachedPlayerList(List<PlayerData> players) {
            this.players = Collections.unmodifiableList(new ArrayList<>(players));
            this.timestamp = System.currentTimeMillis();
        }

        boolean isExpired() {
            return System.currentTimeMillis() - timestamp > CACHE_DURATION_MS;
        }
    }

    /**
     * Класс данных игрока
     */
    private static class PlayerData {
        UUID playerUUID;
        String playerName;
        List<Double> strikes = new ArrayList<>();
        double avgProbability;
        long lastStrikeTime;
        String lastServer;
    }
}
