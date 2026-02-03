package ac.grim.grimac.platform.bukkit.player;

import org.bukkit.configuration.file.FileConfiguration;
import org.bukkit.configuration.file.YamlConfiguration;
import org.bukkit.plugin.java.JavaPlugin;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.logging.Logger;

/**
 * Централизованная система конфигурации для ML
 *
 * PACKAGE: ac.grim.grimac.platform.bukkit.player
 * FOR: Purpur 1.21.1
 *
 * @author ImprovedAImML Team
 * @version 1.0
 */
public class MLConfig {

    private final JavaPlugin plugin;
    private final Logger logger;
    private final File configFile;
    private FileConfiguration config;

    // ========== AI SETTINGS ==========
    private double aiThreshold = 0.85;
    private int aiSequenceLength = 40;
    private int aiStepInterval = 10;
    private double aiBufferMultiplier = 2.0;
    private double aiBufferDecay = 0.5;
    private double aiBufferThreshold = 10.0;

    // ========== HOLOGRAM SETTINGS ==========
    private boolean hologramEnabled = true;
    private boolean hologramOpOnly = true;
    private int hologramMaxStrikes = 8;
    private double hologramHeight = 2.5;
    private boolean hologramShowEmpty = false;

    // ========== GUI SETTINGS ==========
    private boolean guiEnabled = true;
    private int guiItemsPerPage = 28;
    private long guiCacheDuration = 5000;

    // ========== DATASET SETTINGS ==========
    private String datasetPath = "plugins/GrimAC/ml_datasets/";
    private int datasetMinSamples = 500;
    private double datasetBalanceMin = 40.0;
    private double datasetBalanceMax = 60.0;

    // ========== PERFORMANCE SETTINGS ==========
    private int performanceOpCacheRefresh = 100;
    private int performanceHologramUpdate = 10;
    private boolean performanceParallelDataset = true;

    // ========== CHANGE LISTENERS ==========
    private final List<ConfigChangeListener> changeListeners = new ArrayList<>();

    public MLConfig(JavaPlugin plugin) {
        this.plugin = plugin;
        this.logger = plugin.getLogger();
        this.configFile = new File(plugin.getDataFolder(), "ml_config.yml");

        loadConfig();
    }

    /**
     * Загрузить конфигурацию
     */
    public void loadConfig() {
        // Создаем ml_config.yml если не существует
        if (!configFile.exists()) {
            createDefaultConfig();
        }

        config = YamlConfiguration.loadConfiguration(configFile);

        // Load all settings
        loadAllSettings();

        logger.info("[ML Config] Configuration loaded successfully");
    }

    /**
     * Создать конфиг по умолчанию
     */
    private void createDefaultConfig() {
        try {
            plugin.getDataFolder().mkdirs();

            // Создаем конфиг с defaults
            FileConfiguration defaultConfig = new YamlConfiguration();

            // AI section
            defaultConfig.set("ai.threshold", 0.85);
            defaultConfig.set("ai.sequence_length", 40);
            defaultConfig.set("ai.step_interval", 10);
            defaultConfig.set("ai.buffer_multiplier", 2.0);
            defaultConfig.set("ai.buffer_decay", 0.5);
            defaultConfig.set("ai.buffer_threshold", 10.0);

            // Hologram section
            defaultConfig.set("hologram.enabled", true);
            defaultConfig.set("hologram.op_only", true);
            defaultConfig.set("hologram.max_strikes", 8);
            defaultConfig.set("hologram.height", 2.5);
            defaultConfig.set("hologram.show_empty", false);

            // GUI section
            defaultConfig.set("gui.enabled", true);
            defaultConfig.set("gui.items_per_page", 28);
            defaultConfig.set("gui.cache_duration_ms", 5000);

            // Dataset section
            defaultConfig.set("dataset.path", "plugins/GrimAC/ml_datasets/");
            defaultConfig.set("dataset.min_samples", 500);
            defaultConfig.set("dataset.balance_min", 40.0);
            defaultConfig.set("dataset.balance_max", 60.0);

            // Performance section
            defaultConfig.set("performance.op_cache_refresh_ticks", 100);
            defaultConfig.set("performance.hologram_update_ticks", 10);
            defaultConfig.set("performance.parallel_dataset_analysis", true);

            defaultConfig.save(configFile);
            logger.info("[ML Config] Created default ml_config.yml");

        } catch (IOException e) {
            logger.severe("[ML Config] Failed to create default config: " + e.getMessage());
        }
    }

    /**
     * Reload конфигурации
     */
    public void reload() {
        logger.info("[ML Config] Reloading configuration...");

        // Save old values for change detection
        MLConfig oldConfig = this.clone();

        // Reload from file
        config = YamlConfiguration.loadConfiguration(configFile);

        loadAllSettings();

        // Notify listeners
        notifyListeners(oldConfig, this);

        logger.info("[ML Config] Configuration reloaded successfully");
    }

    /**
     * Загрузить все настройки
     */
    private void loadAllSettings() {
        // AI Settings
        aiThreshold = config.getDouble("ai.threshold", 0.85);
        aiSequenceLength = config.getInt("ai.sequence_length", 40);
        aiStepInterval = config.getInt("ai.step_interval", 10);
        aiBufferMultiplier = config.getDouble("ai.buffer_multiplier", 2.0);
        aiBufferDecay = config.getDouble("ai.buffer_decay", 0.5);
        aiBufferThreshold = config.getDouble("ai.buffer_threshold", 10.0);

        // Hologram Settings
        hologramEnabled = config.getBoolean("hologram.enabled", true);
        hologramOpOnly = config.getBoolean("hologram.op_only", true);
        hologramMaxStrikes = config.getInt("hologram.max_strikes", 8);
        hologramHeight = config.getDouble("hologram.height", 2.5);
        hologramShowEmpty = config.getBoolean("hologram.show_empty", false);

        // GUI Settings
        guiEnabled = config.getBoolean("gui.enabled", true);
        guiItemsPerPage = config.getInt("gui.items_per_page", 28);
        guiCacheDuration = config.getLong("gui.cache_duration_ms", 5000);

        // Dataset Settings
        datasetPath = config.getString("dataset.path", "plugins/GrimAC/ml_datasets/");
        datasetMinSamples = config.getInt("dataset.min_samples", 500);
        datasetBalanceMin = config.getDouble("dataset.balance_min", 40.0);
        datasetBalanceMax = config.getDouble("dataset.balance_max", 60.0);

        // Performance Settings
        performanceOpCacheRefresh = config.getInt("performance.op_cache_refresh_ticks", 100);
        performanceHologramUpdate = config.getInt("performance.hologram_update_ticks", 10);
        performanceParallelDataset = config.getBoolean("performance.parallel_dataset_analysis", true);
    }

    /**
     * Сохранить конфигурацию
     */
    public void save() {
        try {
            config.save(configFile);
            logger.info("[ML Config] Configuration saved successfully");
        } catch (IOException e) {
            logger.severe("[ML Config] Failed to save configuration: " + e.getMessage());
        }
    }

    // ========== GETTERS ==========

    public double getAIThreshold() { return aiThreshold; }
    public int getAISequenceLength() { return aiSequenceLength; }
    public int getAIStepInterval() { return aiStepInterval; }
    public double getAIBufferMultiplier() { return aiBufferMultiplier; }
    public double getAIBufferDecay() { return aiBufferDecay; }
    public double getAIBufferThreshold() { return aiBufferThreshold; }

    public boolean isHologramEnabled() { return hologramEnabled; }
    public boolean isHologramOpOnly() { return hologramOpOnly; }
    public int getHologramMaxStrikes() { return hologramMaxStrikes; }
    public double getHologramHeight() { return hologramHeight; }
    public boolean isHologramShowEmpty() { return hologramShowEmpty; }

    public boolean isGUIEnabled() { return guiEnabled; }
    public int getGUIItemsPerPage() { return guiItemsPerPage; }
    public long getGUICacheDuration() { return guiCacheDuration; }

    public String getDatasetPath() { return datasetPath; }
    public int getDatasetMinSamples() { return datasetMinSamples; }
    public double getDatasetBalanceMin() { return datasetBalanceMin; }
    public double getDatasetBalanceMax() { return datasetBalanceMax; }

    public int getPerformanceOpCacheRefresh() { return performanceOpCacheRefresh; }
    public int getPerformanceHologramUpdate() { return performanceHologramUpdate; }
    public boolean isPerformanceParallelDataset() { return performanceParallelDataset; }

    // ========== CHANGE LISTENERS ==========

    public void addChangeListener(ConfigChangeListener listener) {
        changeListeners.add(listener);
    }

    public void removeChangeListener(ConfigChangeListener listener) {
        changeListeners.remove(listener);
    }

    private void notifyListeners(MLConfig oldConfig, MLConfig newConfig) {
        for (ConfigChangeListener listener : changeListeners) {
            listener.onConfigChange(oldConfig, newConfig);
        }
    }

    /**
     * Clone для change detection
     */
    public MLConfig clone() {
        MLConfig cloned = new MLConfig(this.plugin);
        cloned.aiThreshold = this.aiThreshold;
        cloned.aiSequenceLength = this.aiSequenceLength;
        cloned.aiStepInterval = this.aiStepInterval;
        cloned.aiBufferMultiplier = this.aiBufferMultiplier;
        cloned.aiBufferDecay = this.aiBufferDecay;
        cloned.aiBufferThreshold = this.aiBufferThreshold;

        cloned.hologramEnabled = this.hologramEnabled;
        cloned.hologramOpOnly = this.hologramOpOnly;
        cloned.hologramMaxStrikes = this.hologramMaxStrikes;
        cloned.hologramHeight = this.hologramHeight;
        cloned.hologramShowEmpty = this.hologramShowEmpty;

        cloned.guiEnabled = this.guiEnabled;
        cloned.guiItemsPerPage = this.guiItemsPerPage;
        cloned.guiCacheDuration = this.guiCacheDuration;

        cloned.datasetPath = this.datasetPath;
        cloned.datasetMinSamples = this.datasetMinSamples;
        cloned.datasetBalanceMin = this.datasetBalanceMin;
        cloned.datasetBalanceMax = this.datasetBalanceMax;

        cloned.performanceOpCacheRefresh = this.performanceOpCacheRefresh;
        cloned.performanceHologramUpdate = this.performanceHologramUpdate;
        cloned.performanceParallelDataset = this.performanceParallelDataset;

        return cloned;
    }

    /**
     * Интерфейс для change listeners
     */
    public interface ConfigChangeListener {
        void onConfigChange(MLConfig oldConfig, MLConfig newConfig);
    }
}
