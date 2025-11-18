// Modern AI Stock Trading Dashboard - JavaScript
// Advanced functionality with LSTM, TrendSpider, and Polynomial Regression

const API_BASE = 'http://localhost:5050/api';

// Global state - MEMORY OPTIMIZED
let currentStocks = [];
let currentPredictions = [];
let analysisChart = null;
let currentSort = { column: -1, direction: 'asc' };

/* Pagination state */
let paginationState = {
    currentPage: 1,
    perPage: 50,  // DEFAULT: Only 50 stocks at a time
    totalPages: 1,
    totalCount: 0,
    sortBy: 'stock_name',
    sortOrder: 'asc',
    search: ''
};

/* AGGRESSIVE Memory management - Frontend RAM optimization */
function clearMemory() {
    console.log('🧹 Clearing memory...');
    
    /* Clear large arrays - keep only essentials */
    if (currentStocks.length > 50) {
        const old = currentStocks.length;
        currentStocks = currentStocks.slice(-50);  /* Keep only last 50 */
        console.log(`  ✂️ Trimmed currentStocks: ${old} → ${currentStocks.length}`);
    }
    if (currentPredictions.length > 50) {
        const old = currentPredictions.length;
        currentPredictions = currentPredictions.slice(-50);
        console.log(`  ✂️ Trimmed currentPredictions: ${old} → ${currentPredictions.length}`);
    }
    
    /* Destroy old charts */
    if (analysisChart) {
        try {
            analysisChart.destroy();
            analysisChart = null;
            console.log('  🗑️ Destroyed analysisChart');
        } catch (e) {}
    }
    
    /* Force garbage collection hint (Chrome/Edge) */
    if (window.gc && typeof window.gc === 'function') {
        try {
            window.gc();
            console.log('  🗑️ Forced garbage collection');
        } catch (e) {}
    }
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    initializeDashboard();
    loadInitialData();
    setupEventListeners();
    
    /* Memory monitoring */
    if (performance.memory) {
        setInterval(() => {
            const used = (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(1);
            const total = (performance.memory.totalJSHeapSize / 1024 / 1024).toFixed(1);
            console.log(`💾 RAM: ${used}MB / ${total}MB`);
            
            /* Auto-cleanup if RAM usage is high */
            if (performance.memory.usedJSHeapSize > 100 * 1024 * 1024) {  /* 100MB threshold */
                console.warn('⚠️ High memory usage detected, clearing...');
                clearMemory();
            }
        }, 30000);  /* Check every 30 seconds */
    }
});

function initializeDashboard() {
    console.log('🚀 Initializing Advanced AI Stock Trading Dashboard');
    console.log('💾 Memory optimization: Active');
    updateSystemStatus();
}

function updateSystemStatus() {
    // Update status badges
    const systemStatus = document.getElementById('systemStatus');
    const modelStatus = document.getElementById('modelStatus');
    const dataStatus = document.getElementById('dataStatus');
    
    // Check system health
    checkSystemHealth().then(health => {
        if (health.success) {
            systemStatus.innerHTML = '<i class="fas fa-check-circle"></i> System Online';
            systemStatus.className = 'badge success';
            
            modelStatus.innerHTML = `<i class="fas fa-brain"></i> ${health.advanced_ai ? 'Advanced AI Ready' : 'Basic AI Ready'}`;
            modelStatus.className = health.advanced_ai ? 'badge success' : 'badge warning';
        } else {
            systemStatus.innerHTML = '<i class="fas fa-exclamation-circle"></i> System Error';
            systemStatus.className = 'badge error';
        }
    });
}

async function checkSystemHealth() {
    try {
        const response = await axios.get(`${API_BASE}/db/performance-metrics`);
        return {
            success: response.data.success,
            advanced_ai: response.data.metrics?.advanced_ai_available || false,
            database_ready: response.data.metrics?.database_stats?.total_stocks > 0
        };
    } catch (error) {
        console.error('System health check failed:', error);
        return { success: false, advanced_ai: false, database_ready: false };
    }
}

async function loadInitialData() {
    try {
        // Load available stocks for analysis dropdown
        const stocksResponse = await axios.get(`${API_BASE}/stocks`);
        if (stocksResponse.data.success) {
            const stockSelect = document.getElementById('analysisStock');
            /* MEMORY FIX: Build options efficiently */
            const options = ['<option value="">Select a stock...</option>'];
            stocksResponse.data.data.forEach(stock => {
                options.push(`<option value="${stock}">${stock}</option>`);
            });
            stockSelect.innerHTML = options.join('');
            options.length = 0;
        }
        
        // Load dashboard statistics
        loadDashboardStats();
        
        // Load model performance
        loadModelPerformance();
        
    } catch (error) {
        console.error('Failed to load initial data:', error);
        showError('Failed to load initial data');
    }
}

async function loadDashboardStats() {
    try {
        const response = await axios.get(`${API_BASE}/db/performance-metrics`);
        if (response.data.success) {
            const metrics = response.data.metrics;
            
            document.getElementById('totalStocks').textContent = 
                metrics.database_stats?.total_stocks?.toLocaleString() || '0';
            
            document.getElementById('totalPredictions').textContent = 
                metrics.prediction_stats?.total_predictions?.toLocaleString() || '0';
            
            // Calculate average accuracy from validation data
            let avgAccuracy = 0;
            if (metrics.model_validation && metrics.model_validation.length > 0) {
                avgAccuracy = metrics.model_validation.reduce((sum, model) => sum + model.accuracy_rate, 0) / 
                             metrics.model_validation.length;
            }
            document.getElementById('avgAccuracy').textContent = `${avgAccuracy.toFixed(1)}%`;
            
            document.getElementById('topPerformer').textContent = 'TBD';
            
            // Update data status
            const dataStatus = document.getElementById('dataStatus');
            if (metrics.database_stats?.total_stocks > 0) {
                dataStatus.innerHTML = `<i class="fas fa-database"></i> ${metrics.database_stats.total_stocks.toLocaleString()} Stocks`;
                dataStatus.className = 'badge success';
            }
        }
    } catch (error) {
        console.error('Failed to load dashboard stats:', error);
    }
}

async function loadModelPerformance() {
    try {
        const response = await axios.get(`${API_BASE}/db/model-comparison`);
        if (response.data.success && response.data.model_stats) {
            const stats = response.data.model_stats;
            
            // Update model accuracy displays
            Object.entries(stats).forEach(([model, data]) => {
                if (model.includes('lstm') || model.includes('LSTM')) {
                    document.getElementById('lstmAccuracy').textContent = `${data.avg_confidence.toFixed(1)}%`;
                } else if (model.includes('polynomial')) {
                    document.getElementById('polynomialAccuracy').textContent = `${data.avg_confidence.toFixed(1)}%`;
                }
            });
            
            // Default values for TrendSpider (part of advanced ensemble)
            document.getElementById('trendspiderAccuracy').textContent = 'N/A';
        }
        
        // Load detailed model performance
        displayModelPerformance();
        
    } catch (error) {
        console.error('Failed to load model performance:', error);
    }
}

async function displayModelPerformance() {
    const container = document.getElementById('modelPerformance');
    
    try {
        const response = await axios.get(`${API_BASE}/db/model-comparison`);
        if (response.data.success) {
            const stats = response.data.model_stats;
            
            let html = '<div class="grid grid-3">';
            
            Object.entries(stats).forEach(([model, data]) => {
                const modelName = model.replace(/_/g, ' ').toUpperCase();
                const confidence = data.avg_confidence.toFixed(1);
                const change = data.avg_predicted_change.toFixed(2);
                const changeClass = change >= 0 ? 'price-up' : 'price-down';
                
                html += `
                    <div class="card">
                        <h3><i class="fas fa-robot"></i> ${modelName}</h3>
                        <div style="margin: 1rem 0;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: var(--primary);">
                                ${confidence}%
                            </div>
                            <div style="color: var(--gray-600); font-size: 0.875rem;">Avg Confidence</div>
                        </div>
                        <div style="margin: 1rem 0;">
                            <div class="${changeClass}" style="font-size: 1.25rem; font-weight: bold;">
                                ${change > 0 ? '+' : ''}${change}%
                            </div>
                            <div style="color: var(--gray-600); font-size: 0.875rem;">Avg Prediction</div>
                        </div>
                        <div style="color: var(--gray-600); font-size: 0.875rem;">
                            <i class="fas fa-chart-bar"></i> ${data.count} predictions
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            container.innerHTML = html;
        }
    } catch (error) {
        container.innerHTML = '<div class="error-message">Failed to load model performance data</div>';
    }
}

function setupEventListeners() {
    // Stock analysis change handler
    document.getElementById('analysisStock').addEventListener('change', function() {
        if (this.value) {
            analyzeStock();
        }
    });
    
    // Period change handler
    document.getElementById('analysisPeriod').addEventListener('change', function() {
        const stock = document.getElementById('analysisStock').value;
        if (stock) {
            analyzeStock();
        }
    });
    
    // Search box with debounce
    let searchTimeout;
    document.getElementById('stockSearch').addEventListener('input', function() {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            paginationState.search = this.value;
            paginationState.currentPage = 1;
            loadAllStocks();
        }, 500); /* Wait 500ms after user stops typing */
    });
}

// Tab switching
function switchTab(tabName) {
    // Update nav tabs
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Load tab-specific data
    switch(tabName) {
        case 'predictions':
            loadPredictionsData();
            break;
        case 'models':
            loadModelPerformance();
            break;
        case 'validation':
            loadValidationData();
            break;
    }
}

// Database operations
async function syncDatabase() {
    const button = event.target;
    const originalText = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Syncing...';
    
    try {
        showSuccess('Starting database synchronization...');
        const response = await axios.post(`${API_BASE}/db/sync-all`);
        
        if (response.data.success) {
            showSuccess(`✅ Database synced! ${response.data.results.success} stocks processed.`);
            loadDashboardStats();
        } else {
            showError(`Sync failed: ${response.data.error}`);
        }
    } catch (error) {
        showError(`Sync error: ${error.message}`);
    } finally {
        button.disabled = false;
        button.innerHTML = originalText;
    }
}

async function testSync() {
    const button = event.target;
    const originalText = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Testing...';
    
    try {
        const response = await axios.post(`${API_BASE}/db/test-sync`);
        
        if (response.data.success) {
            showSuccess(`
                🧪 Sync Test Successful!<br>
                📊 Stock: ${response.data.test_stock}<br>
                📁 Records: ${response.data.records_saved}<br>
                💾 Database: ${response.data.database_stats.total_records} total records
            `);
        } else {
            showError(`Test failed: ${response.data.error}`);
        }
    } catch (error) {
        showError(`Test error: ${error.message}`);
    } finally {
        button.disabled = false;
        button.innerHTML = originalText;
    }
}

async function loadAllStocks() {
    const button = event.target;
    let originalText, buttonElement;
    
    if (button) {
        buttonElement = button;
        originalText = button.innerHTML;
        button.disabled = true;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
    }
    
    // CRITICAL: Clear memory before loading
    clearMemory();
    
    document.getElementById('stocksLoading').style.display = 'block';
    
    try {
        console.log('🔄 Loading stocks with pagination (memory optimized)...');
        
        /* Build query parameters */
        const params = {
            page: paginationState.currentPage,
            per_page: Math.min(paginationState.perPage, 50),  // MAX 50 per page for memory
            sort_by: paginationState.sortBy,
            sort_order: paginationState.sortOrder
        };
        
        if (paginationState.search) {
            params.search = paginationState.search;
        }
        
        const response = await axios.get(`${API_BASE}/db/all-stocks-fast`, { params });
        
        if (response.data.success) {
            // MEMORY SAFE: Replace, don't append
            currentStocks = response.data.data;
            paginationState.totalCount = response.data.total_count;
            paginationState.totalPages = response.data.total_pages;
            paginationState.currentPage = response.data.page;
            
            console.log(`📋 Page ${paginationState.currentPage}/${paginationState.totalPages}, ${currentStocks.length} stocks`);
            console.log(`💾 Memory: ~${(currentStocks.length * 0.5).toFixed(1)}MB`);
            
            displayStocksTable();
            updatePaginationControls();
            showSuccess(`✅ Loaded ${currentStocks.length} stocks (Page ${paginationState.currentPage}/${paginationState.totalPages})`);
        } else {
            showError(`Failed to load stocks: ${response.data.error}`);
        }
    } catch (error) {
        if (error.response?.status === 500) {
            showError('Database not initialized. Please sync database first.');
        } else {
            showError(`Load error: ${error.message}`);
        }
    } finally {
        document.getElementById('stocksLoading').style.display = 'none';
        if (buttonElement) {
            buttonElement.disabled = false;
            buttonElement.innerHTML = originalText;
        }
    }
}

async function updateTodayData() {
    const button = event.target;
    const originalText = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Updating...';
    
    try {
        const response = await axios.post(`${API_BASE}/db/update-today`);
        
        if (response.data.success) {
            showSuccess(`✅ Updated ${response.data.results.success} stocks with today's data`);
            if (currentStocks.length > 0) {
                loadAllStocks(); // Refresh the display
            }
        } else {
            showError(`Update failed: ${response.data.error}`);
        }
    } catch (error) {
        showError(`Update error: ${error.message}`);
    } finally {
        button.disabled = false;
        button.innerHTML = originalText;
    }
}

async function generateBulkPredictions() {
    const modelType = document.getElementById('predictionModel')?.value || 'advanced_ensemble';
    const limit = parseInt(document.getElementById('predictionLimit')?.value || '50');
    const forceRefresh = document.getElementById('forceRefresh')?.checked || false;
    const useAdvancedAI = document.getElementById('useAdvancedAI')?.checked !== false;
    
    await generateAdvancedPredictions(modelType, limit, forceRefresh, useAdvancedAI);
}

async function generateAdvancedPredictions(modelType = 'advanced_ensemble', limit = 50, forceRefresh = false, useAdvancedAI = true) {
    const button = event.target;
    const originalText = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
    
    // Show progress
    const progress = document.getElementById('predictionProgress');
    const progressBar = document.getElementById('predictionProgressBar');
    const status = document.getElementById('predictionStatus');
    
    progress.style.display = 'block';
    progressBar.style.width = '0%';
    status.textContent = 'Initializing AI models...';
    
    try {
        const requestData = {
            limit: limit,
            force_refresh: forceRefresh,
            use_advanced_ai: useAdvancedAI,
            model_type: modelType
        };
        
        status.textContent = `Training ${modelType.replace('_', ' ')} model...`;
        progressBar.style.width = '30%';
        
        const response = await axios.post(`${API_BASE}/db/predict-all`, requestData);
        
        progressBar.style.width = '100%';
        status.textContent = 'Predictions generated successfully!';
        
        if (response.data.success) {
            const results = response.data.results;
            const avgAccuracy = response.data.average_accuracy || 0;
            
            showSuccess(`
                🔮 AI Predictions Generated!<br>
                ✅ Success: ${results.success} stocks<br>
                ⏭️ Skipped: ${results.skipped}<br>
                ❌ Failed: ${results.failed}<br>
                🎯 Average Accuracy: ${avgAccuracy.toFixed(1)}%<br>
                🤖 Model: ${response.data.model_type}<br>
                📅 Date: ${response.data.prediction_date}
            `);
            
            // Load predictions data
            loadPredictionsData();
            loadDashboardStats();
        } else {
            showError(`Prediction failed: ${response.data.error}`);
        }
    } catch (error) {
        showError(`Prediction error: ${error.message}`);
        status.textContent = 'Prediction failed';
    } finally {
        setTimeout(() => {
            progress.style.display = 'none';
        }, 3000);
        
        button.disabled = false;
        button.innerHTML = originalText;
    }
}

async function loadPredictionsData() {
    try {
        const response = await axios.get(`${API_BASE}/db/all-stocks-fast`);
        
        if (response.data.success) {
            currentPredictions = response.data.data.filter(stock => stock.predicted_price !== null);
            displayPredictionsTable();
        }
    } catch (error) {
        console.error('Failed to load predictions:', error);
    }
}

function displayStocksTable() {
    const tbody = document.getElementById('stocksTableBody');
    
    if (!tbody) {
        console.error('❌ stocksTableBody element not found!');
        return;
    }
    
    if (currentStocks.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; padding: 3rem;"><i class="fas fa-info-circle"></i> No stocks data available</td></tr>';
        return;
    }
    
    try {
        // MEMORY OPTIMIZATION: Build HTML efficiently
        const rows = [];
        for (let i = 0; i < currentStocks.length; i++) {
            const stock = currentStocks[i];
            const priceClass = stock.today_pct > 0 ? 'price-up' : stock.today_pct < 0 ? 'price-down' : 'price-neutral';
            const predClass = stock.predicted_change_pct > 0 ? 'price-up' : stock.predicted_change_pct < 0 ? 'price-down' : 'price-neutral';
            
            rows.push(`
                <tr onclick="selectStock('${stock.stock_name}')" style="cursor: pointer;">
                    <td><strong>${stock.stock_name}</strong></td>
                    <td>₹${stock.today_close.toFixed(2)}</td>
                    <td class="${priceClass}">${stock.today_pct >= 0 ? '+' : ''}${stock.today_pct.toFixed(2)}%</td>
                    <td class="${stock.week_7day_pct >= 0 ? 'price-up' : 'price-down'}">${stock.week_7day_pct >= 0 ? '+' : ''}${stock.week_7day_pct.toFixed(2)}%</td>
                    <td>${stock.predicted_price ? '₹' + stock.predicted_price.toFixed(2) : '--'}</td>
                    <td class="${stock.predicted_change_pct ? predClass : ''}">${stock.predicted_change_pct ? (stock.predicted_change_pct >= 0 ? '+' : '') + stock.predicted_change_pct.toFixed(2) + '%' : '--'}</td>
                    <td><span class="badge info">${getModelBadge(stock.model_used || 'N/A')}</span></td>
                    <td>${stock.confidence ? stock.confidence.toFixed(1) + '%' : '--'}</td>
                    <td>${stock.volume ? stock.volume.toLocaleString() : '--'}</td>
                </tr>
            `);
        }
        
        tbody.innerHTML = rows.join('');
        
        // Clear temp array
        rows.length = 0;
        
    } catch (error) {
        console.error('❌ Error updating table:', error);
        showError('Error displaying stocks table: ' + error.message);
    }
}

function displayPredictionsTable() {
    const tbody = document.getElementById('predictionsTableBody');
    
    if (!tbody) return;
    
    if (currentPredictions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 3rem;"><i class="fas fa-info-circle"></i> No predictions available</td></tr>';
        return;
    }
    
    /* MEMORY OPTIMIZATION: Build HTML efficiently without map() */
    const rows = [];
    for (let i = 0; i < currentPredictions.length; i++) {
        const stock = currentPredictions[i];
        const predClass = stock.predicted_change_pct > 0 ? 'price-up' : stock.predicted_change_pct < 0 ? 'price-down' : 'price-neutral';
        const recommendation = getRecommendation(stock.predicted_change_pct, stock.confidence);
        
        rows.push(`
            <tr onclick="selectStock('${stock.stock_name}')" style="cursor: pointer;">
                <td><strong>${stock.stock_name}</strong></td>
                <td>₹${stock.today_close.toFixed(2)}</td>
                <td>₹${stock.predicted_price.toFixed(2)}</td>
                <td class="${predClass}"><strong>${stock.predicted_change_pct >= 0 ? '+' : ''}${stock.predicted_change_pct.toFixed(2)}%</strong></td>
                <td>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="width: 60px; height: 6px; background: var(--gray-200); border-radius: 3px;">
                            <div style="width: ${stock.confidence}%; height: 100%; background: var(--primary); border-radius: 3px;"></div>
                        </div>
                        ${stock.confidence.toFixed(1)}%
                    </div>
                </td>
                <td><span class="badge info">${getModelBadge(stock.model_used)}</span></td>
                <td><span class="badge ${recommendation.class}">${recommendation.text}</span></td>
            </tr>
        `);
    }
    
    tbody.innerHTML = rows.join('');
    rows.length = 0;  /* Clear temp array */
}

function getModelBadge(model) {
    if (!model || model === 'N/A') return 'N/A';
    
    const modelMap = {
        'Advanced_advanced_ensemble': '🎯 Ensemble',
        'Advanced_lstm': '🧠 LSTM',
        'Advanced_polynomial': '📈 Polynomial',
        'Simple_MA_Trend': '📊 MA Trend',
        'MA_Trend': '📊 Simple'
    };
    
    return modelMap[model] || model.replace(/_/g, ' ');
}

function getRecommendation(changePct, confidence) {
    if (changePct > 3 && confidence > 80) return { text: 'Strong Buy', class: 'success' };
    if (changePct > 1 && confidence > 70) return { text: 'Buy', class: 'success' };
    if (changePct < -3 && confidence > 80) return { text: 'Strong Sell', class: 'error' };
    if (changePct < -1 && confidence > 70) return { text: 'Sell', class: 'warning' };
    return { text: 'Hold', class: 'info' };
}

// Individual stock analysis
async function analyzeStock() {
    const stockName = document.getElementById('analysisStock').value;
    const period = document.getElementById('analysisPeriod').value;
    
    if (!stockName) return;
    
    const container = document.getElementById('stockAnalysis');
    container.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Analyzing stock...</div>';
    
    try {
        const response = await axios.get(`${API_BASE}/stock/${stockName}`, {
            params: { period }
        });
        
        if (response.data.success) {
            const summary = response.data.summary;
            const data = response.data.data;
            
            // Display analysis
            const changeClass = summary.change >= 0 ? 'price-up' : 'price-down';
            
            container.innerHTML = `
                <div style="margin-top: 1rem;">
                    <h4>${stockName} Analysis</h4>
                    <div class="grid grid-2" style="margin-top: 1rem;">
                        <div>
                            <div style="font-size: 1.5rem; font-weight: bold;">₹${summary.current_price.toFixed(2)}</div>
                            <div class="${changeClass}" style="font-size: 1rem; margin: 0.5rem 0;">
                                ${summary.change >= 0 ? '+' : ''}₹${summary.change.toFixed(2)} 
                                (${summary.change_pct >= 0 ? '+' : ''}${summary.change_pct.toFixed(2)}%)
                            </div>
                        </div>
                        <div>
                            <div style="font-size: 0.875rem; color: var(--gray-600);">52W Range</div>
                            <div style="font-size: 0.875rem;">₹${summary.low_52w.toFixed(2)} - ₹${summary.high_52w.toFixed(2)}</div>
                        </div>
                    </div>
                </div>
            `;
            
            // Draw chart
            drawAnalysisChart(data, stockName);
            
            // Generate AI prediction for this stock
            generateSinglePrediction(stockName);
            
        } else {
            container.innerHTML = '<div class="error-message">Failed to load stock data</div>';
        }
    } catch (error) {
        container.innerHTML = '<div class="error-message">Error loading stock analysis</div>';
    }
}

async function generateSinglePrediction(stockName) {
    try {
        const response = await axios.get(`${API_BASE}/predict/${stockName}`);
        
        if (response.data.success) {
            const prediction = response.data.prediction;
            const signal = response.data.signal;
            
            const container = document.getElementById('stockAnalysis');
            const predClass = prediction.change_pct >= 0 ? 'price-up' : 'price-down';
            
            container.innerHTML += `
                <div style="margin-top: 1.5rem; padding: 1rem; background: var(--gray-50); border-radius: 0.5rem;">
                    <h5><i class="fas fa-brain"></i> AI Prediction</h5>
                    <div class="grid grid-2" style="margin-top: 1rem;">
                        <div>
                            <div style="font-size: 1.25rem; font-weight: bold;">₹${prediction.predicted_price.toFixed(2)}</div>
                            <div class="${predClass}" style="font-size: 0.875rem;">
                                ${prediction.change_pct >= 0 ? '+' : ''}${prediction.change_pct.toFixed(2)}% expected
                            </div>
                        </div>
                        <div>
                            <div style="font-size: 0.875rem; color: var(--gray-600);">Confidence</div>
                            <div style="font-size: 1rem; font-weight: bold; color: var(--primary);">
                                ${prediction.confidence.toFixed(1)}%
                            </div>
                        </div>
                    </div>
                    <div style="margin-top: 1rem;">
                        <span class="badge ${signal.signal.includes('BUY') ? 'success' : signal.signal.includes('SELL') ? 'error' : 'info'}">
                            ${signal.signal}
                        </span>
                    </div>
                </div>
            `;
        }
    } catch (error) {
        console.error('Single prediction failed:', error);
    }
}

function drawAnalysisChart(data, stockName) {
    const canvas = document.getElementById('analysisChart');
    const ctx = canvas.getContext('2d');
    
    if (analysisChart) {
        analysisChart.destroy();
    }
    
    const dates = data.map(d => new Date(d.Date).toLocaleDateString());
    const prices = data.map(d => d.Close);
    const volumes = data.map(d => d.Volume);
    
    analysisChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Price',
                    data: prices,
                    borderColor: 'rgb(79, 70, 229)',
                    backgroundColor: 'rgba(79, 70, 229, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Volume',
                    data: volumes,
                    type: 'bar',
                    backgroundColor: 'rgba(6, 182, 212, 0.3)',
                    borderColor: 'rgb(6, 182, 212)',
                    borderWidth: 1,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `${stockName} - Price & Volume Analysis`
                },
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Price (₹)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Volume'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

// Validation functions
async function validatePredictions() {
    const container = document.getElementById('validationResults');
    container.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Running validation...</div>';
    
    try {
        const response = await axios.post(`${API_BASE}/db/validate-predictions`);
        
        if (response.data.success) {
            const validations = response.data.validations;
            const summary = response.data.summary;
            
            if (validations.length === 0) {
                container.innerHTML = '<div class="error-message">No predictions available for validation</div>';
                return;
            }
            
            let html = `
                <div class="grid grid-4" style="margin-bottom: 2rem;">
                    <div class="stat-card">
                        <div class="stat-value">${summary.total_validated}</div>
                        <div class="stat-label">Validated</div>
                    </div>
                    <div class="stat-card" style="background: linear-gradient(135deg, var(--success) 0%, #059669 100%);">
                        <div class="stat-value">${summary.average_accuracy.toFixed(1)}%</div>
                        <div class="stat-label">Price Accuracy</div>
                    </div>
                    <div class="stat-card" style="background: linear-gradient(135deg, var(--secondary) 0%, #0891b2 100%);">
                        <div class="stat-value">${summary.direction_accuracy.toFixed(1)}%</div>
                        <div class="stat-label">Direction Accuracy</div>
                    </div>
                    <div class="stat-card" style="background: linear-gradient(135deg, var(--warning) 0%, #d97706 100%);">
                        <div class="stat-value">${summary.average_price_error_pct.toFixed(1)}%</div>
                        <div class="stat-label">Avg Error</div>
                    </div>
                </div>
            `;
            
            html += '<div style="overflow-x: auto;"><table class="modern-table"><thead><tr>';
            html += '<th>Stock</th><th>Predicted</th><th>Actual</th><th>Error %</th><th>Direction</th><th>Model</th><th>Confidence</th>';
            html += '</tr></thead><tbody>';
            
            validations.forEach(v => {
                const errorClass = v.price_error_pct < 5 ? 'price-up' : v.price_error_pct < 10 ? 'price-neutral' : 'price-down';
                const directionIcon = v.direction_correct ? '✅' : '❌';
                
                html += `
                    <tr>
                        <td><strong>${v.stock_name}</strong></td>
                        <td>₹${v.predicted_price.toFixed(2)}</td>
                        <td>₹${v.actual_price.toFixed(2)}</td>
                        <td class="${errorClass}">${v.price_error_pct.toFixed(2)}%</td>
                        <td>${directionIcon}</td>
                        <td><span class="badge info">${getModelBadge(v.model_used)}</span></td>
                        <td>${v.confidence.toFixed(1)}%</td>
                    </tr>
                `;
            });
            
            html += '</tbody></table></div>';
            container.innerHTML = html;
            
        } else {
            container.innerHTML = '<div class="error-message">Validation failed</div>';
        }
    } catch (error) {
        container.innerHTML = '<div class="error-message">Error running validation</div>';
    }
}

async function loadValidationData() {
    // Auto-run validation when tab is opened
    validatePredictions();
}

/* Pagination control functions */
function updatePaginationControls() {
    const controls = document.getElementById('paginationControls');
    const pageInfo = document.getElementById('pageInfo');
    const totalInfo = document.getElementById('totalStocksInfo');
    const firstBtn = document.getElementById('firstPageBtn');
    const prevBtn = document.getElementById('prevPageBtn');
    const nextBtn = document.getElementById('nextPageBtn');
    const lastBtn = document.getElementById('lastPageBtn');
    
    controls.style.display = 'block';
    pageInfo.textContent = `Page ${paginationState.currentPage} of ${paginationState.totalPages}`;
    totalInfo.textContent = `Total: ${paginationState.totalCount.toLocaleString()} stocks`;
    
    /* Disable buttons at boundaries */
    firstBtn.disabled = paginationState.currentPage === 1;
    prevBtn.disabled = paginationState.currentPage === 1;
    nextBtn.disabled = paginationState.currentPage === paginationState.totalPages;
    lastBtn.disabled = paginationState.currentPage === paginationState.totalPages;
}

function loadPage(page) {
    paginationState.currentPage = page;
    loadAllStocks();
}

function loadNextPage() {
    if (paginationState.currentPage < paginationState.totalPages) {
        paginationState.currentPage++;
        loadAllStocks();
    }
}

function loadPrevPage() {
    if (paginationState.currentPage > 1) {
        paginationState.currentPage--;
        loadAllStocks();
    }
}

function loadLastPage() {
    paginationState.currentPage = paginationState.totalPages;
    loadAllStocks();
}

function changePerPage() {
    paginationState.perPage = parseInt(document.getElementById('perPageSelect').value);
    paginationState.currentPage = 1;
    loadAllStocks();
}

/* Server-side table sorting */
function sortTableServer(column) {
    /* Toggle sort order if clicking same column */
    if (paginationState.sortBy === column) {
        paginationState.sortOrder = paginationState.sortOrder === 'asc' ? 'desc' : 'asc';
    } else {
        paginationState.sortBy = column;
        paginationState.sortOrder = 'desc';
    }
    
    paginationState.currentPage = 1; /* Reset to first page when sorting */
    loadAllStocks();
}

/* Legacy client-side sorting - kept for backwards compatibility */
function sortTable(columnIndex) {
    /* This function is deprecated, use sortTableServer instead */
    console.warn('sortTable() is deprecated, use sortTableServer() instead');
}

function sortPredictionsTable(columnIndex) {
    // Similar sorting logic for predictions table
    currentPredictions.sort((a, b) => {
        let aVal, bVal;
        
        switch(columnIndex) {
            case 0: aVal = a.stock_name; bVal = b.stock_name; break;
            case 1: aVal = a.today_close; bVal = b.today_close; break;
            case 2: aVal = a.predicted_price; bVal = b.predicted_price; break;
            case 3: aVal = a.predicted_change_pct; bVal = b.predicted_change_pct; break;
            case 4: aVal = a.confidence; bVal = b.confidence; break;
            case 5: aVal = a.model_used; bVal = b.model_used; break;
            default: return 0;
        }
        
        if (typeof aVal === 'string') {
            return aVal.localeCompare(bVal);
        } else {
            return bVal - aVal; // Always desc for predictions
        }
    });
    
    displayPredictionsTable();
}

// Utility functions
function selectStock(stockName) {
    // Switch to analysis tab and select the stock
    switchTab('analysis');
    document.getElementById('analysisStock').value = stockName;
    analyzeStock();
}

function refreshStocksList() {
    loadAllStocks();
}

function exportStocksData() {
    if (currentStocks.length === 0) {
        showError('No data to export. Load stocks first.');
        return;
    }
    
    const csvData = convertToCSV(currentStocks);
    const blob = new Blob([csvData], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `stocks_data_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

function convertToCSV(data) {
    const headers = Object.keys(data[0]);
    const rows = data.map(row => headers.map(header => row[header] || '').join(','));
    return [headers.join(','), ...rows].join('\n');
}

function showError(message) {
    console.error('🔴 Error:', message);
    
    // Create temporary error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = message;
    errorDiv.style.position = 'fixed';
    errorDiv.style.top = '20px';
    errorDiv.style.right = '20px';
    errorDiv.style.zIndex = '9999';
    errorDiv.style.maxWidth = '400px';
    document.body.appendChild(errorDiv);
    
    setTimeout(() => {
        if (document.body.contains(errorDiv)) {
            document.body.removeChild(errorDiv);
        }
    }, 5000);
}

function showSuccess(message) {
    console.log('🟢 Success:', message);
    
    // Create temporary success message  
    const successDiv = document.createElement('div');
    successDiv.className = 'success-message';
    successDiv.innerHTML = message;
    successDiv.style.position = 'fixed';
    successDiv.style.top = '20px';
    successDiv.style.right = '20px';
    successDiv.style.zIndex = '9999';
    successDiv.style.maxWidth = '400px';
    document.body.appendChild(successDiv);
    
    setTimeout(() => {
        if (document.body.contains(successDiv)) {
            document.body.removeChild(successDiv);
        }
    }, 5000);
}

// Debug function to test data loading
async function debugLoadStocks() {
    console.log('🐛 DEBUG: Starting debug load test...');
    
    try {
        // Test API endpoint directly
        console.log('🔗 Testing API endpoint:', `${API_BASE}/db/all-stocks-fast`);
        
        const response = await fetch(`${API_BASE}/db/all-stocks-fast`);
        console.log('📡 Fetch response:', response);
        console.log('✅ Response status:', response.status, response.statusText);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('📊 Parsed JSON data:', data);
        console.log('🎯 Data success:', data.success);
        console.log('📈 Data count:', data.count);
        console.log('📋 First 3 stocks:', data.data?.slice(0, 3));
        
        if (data.success && data.data) {
            currentStocks = data.data;
            console.log('✅ currentStocks set to:', currentStocks.length, 'stocks');
            
            // Test table display
            const tbody = document.getElementById('stocksTableBody');
            console.log('🎯 Table body element found:', !!tbody);
            
            if (tbody) {
                displayStocksTable();
                showSuccess(`🐛 Debug: Successfully loaded and displayed ${currentStocks.length} stocks!`);
            } else {
                showError('🐛 Debug: Table body element not found!');
            }
        } else {
            showError('🐛 Debug: API returned unsuccessful response');
        }
        
    } catch (error) {
        console.error('🐛 Debug error:', error);
        showError(`🐛 Debug failed: ${error.message}`);
    }
}

/* Universal AI Training Functions */
async function trainUniversalModel() {
    const button = event.target;
    const originalText = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
    
    const statusElement = document.getElementById('universalModelStatus');
    statusElement.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training on all stocks... Please wait';
    
    try {
        showSuccess('🌍 Starting universal model training on all stocks...');
        
        const response = await axios.post(`${API_BASE}/db/train-universal`, {
            max_stocks: null  /* Train on all stocks - backend can handle it */
        });
        
        if (response.data.success) {
            const models = response.data.models;
            let message = '🎉 Universal Model Training Complete!<br><br>';
            
            for (const [modelName, metrics] of Object.entries(models)) {
                message += `<strong>${modelName.replace('_', ' ').toUpperCase()}</strong><br>`;
                message += `  Direction Accuracy: ${metrics.direction_accuracy.toFixed(2)}%<br>`;
                message += `  R² Score: ${metrics.test_r2.toFixed(4)}<br><br>`;
            }
            
            statusElement.innerHTML = 
                '<i class="fas fa-check-circle"></i> Universal model trained successfully!';
            
            showSuccess(message);
        } else {
            statusElement.innerHTML = '<i class="fas fa-exclamation-circle"></i> Training failed';
            showError('Training failed: ' + response.data.error);
        }
    } catch (error) {
        statusElement.innerHTML = '<i class="fas fa-exclamation-circle"></i> Error';
        showError('Training error: ' + error.message);
    } finally {
        button.disabled = false;
        button.innerHTML = originalText;
    }
}

async function generateUniversalPredictions() {
    const button = event.target;
    const originalText = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';
    
    try {
        showSuccess('🔮 Generating predictions using universal model...');
        
        const response = await axios.post(`${API_BASE}/db/predict-all-universal`, {
            limit: 5000
        });
        
        if (response.data.success) {
            const results = response.data.results;
            
            showSuccess(`
                🎯 Universal Predictions Complete!<br>
                ✅ Success: ${results.success} stocks<br>
                ❌ Failed: ${results.failed}<br>
                📅 Date: ${response.data.prediction_date}<br><br>
                <small>These predictions use a model trained on ALL stocks combined!</small>
            `);
            
            /* Refresh stocks list to show new predictions */
            if (currentStocks.length > 0) {
                loadAllStocks();
            }
            
            loadDashboardStats();
        } else {
            showError('Prediction failed: ' + response.data.error);
        }
    } catch (error) {
        if (error.response?.status === 400) {
            showError('Universal model not trained yet. Please train the model first.');
        } else {
            showError('Prediction error: ' + error.message);
        }
    } finally {
        button.disabled = false;
        button.innerHTML = originalText;
    }
}

// Auto-refresh data every 5 minutes
setInterval(() => {
    if (currentStocks.length > 0) {
        loadDashboardStats();
    }
}, 5 * 60 * 1000);
