// AI Stock Trading Dashboard - Frontend JavaScript

const API_BASE = 'http://localhost:5050/api';

let currentStock = null;
let priceChart = null;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    loadStocks();
    loadStrategies();
});

// Load available stocks
async function loadStocks() {
    try {
        const response = await axios.get(`${API_BASE}/stocks`);
        const stocks = response.data.data;
        
        const select = document.getElementById('stockSelect');
        select.innerHTML = '<option value="">Select a stock...</option>';
        
        stocks.forEach(stock => {
            const option = document.createElement('option');
            option.value = stock;
            option.textContent = stock;
            select.appendChild(option);
        });
        
    } catch (error) {
        showError('Failed to load stocks: ' + error.message);
    }
}

// Load stock data
async function loadStockData() {
    const stockName = document.getElementById('stockSelect').value;
    const period = document.getElementById('periodSelect').value;
    
    if (!stockName) return;
    
    currentStock = stockName;
    hideError();
    
    try {
        console.log('Loading stock data for:', stockName, 'period:', period);
        
        const response = await axios.get(`${API_BASE}/stock/${stockName}`, {
            params: { period }
        });
        
        console.log('API Response:', response.data);
        console.log('Response type:', typeof response.data);
        console.log('Response.data.success:', response.data.success);
        console.log('Response.data.data:', response.data.data);
        console.log('Response.data.summary:', response.data.summary);
        
        if (!response.data || !response.data.success) {
            console.error('Validation failed: response.data =', response.data, 'response.data.success =', response.data.success);
            throw new Error(response.data?.error || 'Failed to load data');
        }
        
        const data = response.data.data;
        const summary = response.data.summary;
        
        console.log('Extracted data:', data, 'length:', data?.length);
        console.log('Extracted summary:', summary);
        
        // Validate data
        if (!data || data.length === 0) {
            console.error('Data validation failed:', {data, length: data?.length});
            throw new Error('No data available for this stock');
        }
        
        if (!summary) {
            console.error('Summary validation failed:', summary);
            throw new Error('Summary data not available');
        }
        
        console.log('Data validated. Records:', data.length, 'Summary:', summary);
        
        // Display current metrics
        try {
            displayCurrentMetrics(summary);
            console.log('Metrics displayed successfully');
        } catch (metricsError) {
            console.error('Error displaying metrics:', metricsError);
            throw new Error('Failed to display metrics: ' + metricsError.message);
        }
        
        // Draw chart
        try {
            drawPriceChart(data);
            console.log('Chart drawn successfully');
        } catch (chartError) {
            console.error('Error drawing chart:', chartError);
            throw new Error('Failed to draw chart: ' + chartError.message);
        }
        
    } catch (error) {
        showError('Failed to load stock data: ' + error.message);
        console.error('Stock data error:', error);
        console.error('Error stack:', error.stack);
    }
}

// Display current metrics
function displayCurrentMetrics(summary) {
    const container = document.getElementById('currentMetrics');
    
    // Safety checks
    if (!container) {
        console.error('currentMetrics container not found');
        throw new Error('currentMetrics container not found in DOM');
    }
    
    if (!summary) {
        container.innerHTML = '<div class="error">Summary data not available</div>';
        return;
    }
    
    // Safe value extraction with defaults
    const currentPrice = Number(summary.current_price) || 0;
    const change = Number(summary.change) || 0;
    const changePct = Number(summary.change_pct) || 0;
    const highPrice = Number(summary.high_price) || 0;
    const lowPrice = Number(summary.low_price) || 0;
    const high52w = Number(summary.high_52w) || 0;
    const low52w = Number(summary.low_52w) || 0;
    
    const changeClass = change >= 0 ? 'positive' : 'negative';
    const changeSymbol = change >= 0 ? '+' : '';
    
    container.innerHTML = `
        <div class="metric">
            <span class="metric-label">Current Price</span>
            <span class="metric-value">₹${currentPrice.toFixed(2)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Change</span>
            <span class="metric-value ${changeClass}">
                ${changeSymbol}₹${change.toFixed(2)} 
                (${changeSymbol}${changePct.toFixed(2)}%)
            </span>
        </div>
        <div class="metric">
            <span class="metric-label">High</span>
            <span class="metric-value">₹${highPrice.toFixed(2)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Low</span>
            <span class="metric-value">₹${lowPrice.toFixed(2)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">52W High</span>
            <span class="metric-value">₹${high52w.toFixed(2)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">52W Low</span>
            <span class="metric-value">₹${low52w.toFixed(2)}</span>
        </div>
    `;
}

// Draw price chart
function drawPriceChart(data) {
    const canvas = document.getElementById('priceChart');
    
    if (!canvas) {
        console.error('priceChart canvas not found');
        throw new Error('priceChart canvas not found in DOM');
    }
    
    const ctx = canvas.getContext('2d');
    
    // Destroy existing chart
    if (priceChart) {
        priceChart.destroy();
    }
    
    // Helper function to handle NaN and null values
    const cleanValue = (val) => (val === null || isNaN(val) || val === undefined) ? null : val;
    
    const dates = data.map(d => {
        // Handle different date formats
        if (typeof d.Date === 'string') {
            return d.Date.includes('GMT') ? new Date(d.Date).toLocaleDateString() : d.Date.split('T')[0];
        }
        return d.Date;
    });
    const prices = data.map(d => cleanValue(d.Close));
    const ma20 = data.map(d => cleanValue(d.MA_20));
    const ma50 = data.map(d => cleanValue(d.MA_50));
    
    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Close Price',
                    data: prices,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    spanGaps: true
                },
                {
                    label: 'MA 20',
                    data: ma20,
                    borderColor: '#10b981',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.4,
                    spanGaps: true
                },
                {
                    label: 'MA 50',
                    data: ma50,
                    borderColor: '#f59e0b',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.4,
                    spanGaps: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null && !isNaN(context.parsed.y)) {
                                label += '₹' + context.parsed.y.toFixed(2);
                            } else {
                                label += 'N/A';
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 10
                    }
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        callback: function(value) {
                            return '₹' + value.toFixed(0);
                        }
                    }
                }
            }
        }
    });
}

// Get AI prediction
async function getPrediction() {
    if (!currentStock) {
        showError('Please select a stock first');
        return;
    }
    
    hideError();
    
    // Show loading
    document.getElementById('predictionMetrics').innerHTML = '<div class="loading">Generating AI prediction...</div>';
    document.getElementById('tradingSignal').innerHTML = '<div class="loading">Analyzing...</div>';
    document.getElementById('recommendations').innerHTML = '<div class="loading">Computing recommendations...</div>';
    
    try {
        const response = await axios.get(`${API_BASE}/predict/${currentStock}`);
        const result = response.data;
        
        // Display prediction
        displayPrediction(result.prediction);
        
        // Display signal
        displaySignal(result.signal);
        
        // Display recommendations
        displayRecommendations(result.recommendations);
        
    } catch (error) {
        showError('Failed to generate prediction: ' + error.message);
        document.getElementById('predictionMetrics').innerHTML = '<div class="error">Prediction failed</div>';
    }
}

// Display prediction
function displayPrediction(prediction) {
    const container = document.getElementById('predictionMetrics');
    
    const changeClass = prediction.change_pct >= 0 ? 'positive' : 'negative';
    const changeSymbol = prediction.change_pct >= 0 ? '+' : '';
    
    container.innerHTML = `
        <div class="metric">
            <span class="metric-label">Predicted Price</span>
            <span class="metric-value">₹${Number(prediction.predicted_price || 0).toFixed(2)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Expected Change</span>
            <span class="metric-value ${changeClass}">
                ${changeSymbol}₹${Number(prediction.change || 0).toFixed(2)} 
                (${changeSymbol}${Number(prediction.change_pct || 0).toFixed(2)}%)
            </span>
        </div>
        <div class="metric">
            <span class="metric-label">Confidence</span>
            <span class="metric-value">${Number(prediction.confidence || 0).toFixed(2)}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Model</span>
            <span class="metric-value">${(prediction.model || 'ensemble').toUpperCase()}</span>
        </div>
    `;
}

// Display trading signal
function displaySignal(signal) {
    if (!signal) return;
    
    const container = document.getElementById('tradingSignal');
    
    let signalClass = 'signal-hold';
    if (signal.signal.includes('BUY')) signalClass = 'signal-buy';
    if (signal.signal.includes('SELL')) signalClass = 'signal-sell';
    
    container.innerHTML = `
        <div style="text-align: center; margin-bottom: 20px;">
            <span class="signal-badge ${signalClass}">${signal.signal}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Action</span>
            <span class="metric-value">${signal.action}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Expected Return</span>
            <span class="metric-value">${signal.expected_change >= 0 ? '+' : ''}${signal.expected_change.toFixed(2)}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Confidence</span>
            <span class="metric-value">${signal.confidence.toFixed(2)}%</span>
        </div>
    `;
}

// Display recommendations
function displayRecommendations(recommendations) {
    const container = document.getElementById('recommendations');
    
    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = '<p>No recommendations available</p>';
        return;
    }
    
    container.innerHTML = recommendations.map(rec => `
        <div class="recommendation-item">
            <h3 style="color: #667eea; margin-bottom: 10px;">${rec.strategy.toUpperCase()} Strategy</h3>
            <p><strong>Action:</strong> ${rec.action}</p>
            <p><strong>Reason:</strong> ${rec.reason}</p>
            <p><strong>Position Size:</strong> ${(rec.position_size * 100).toFixed(0)}% of capital</p>
            <p><strong>Stop Loss:</strong> ${rec.stop_loss}%</p>
            <p><strong>Take Profit:</strong> ${rec.take_profit}%</p>
            <p><strong>Confidence:</strong> ${rec.confidence.toFixed(2)}%</p>
        </div>
    `).join('');
}

// Validate predictions
async function validatePredictions() {
    if (!currentStock) {
        showError('Please select a stock first');
        return;
    }
    
    hideError();
    
    try {
        const response = await axios.get(`${API_BASE}/validate/${currentStock}`);
        const result = response.data;
        
        if (result.validations.length === 0) {
            showError('No predictions to validate yet. Make some predictions first!');
            return;
        }
        
        // Show validation results
        alert(`Validation Results:\n\n` +
              `Validated: ${result.validations.length} predictions\n` +
              `Average Accuracy: ${result.report.avg_accuracy.toFixed(2)}%\n` +
              `Accurate: ${result.report.accurate_count}\n` +
              `Moderate: ${result.report.moderate_count}\n` +
              `Poor: ${result.report.poor_count}`);
        
    } catch (error) {
        showError('Failed to validate: ' + error.message);
    }
}

// Filter stocks
async function filterStocks() {
    hideError();
    
    const criteria = {
        price_min: parseFloat(document.getElementById('filterPriceMin').value) || undefined,
        price_max: parseFloat(document.getElementById('filterPriceMax').value) || undefined,
        returns_1d_min: parseFloat(document.getElementById('filterReturns1d').value) || undefined,
        sort_by: document.getElementById('filterSortBy').value,
        sort_order: 'desc',
        limit: 50
    };
    
    // Remove undefined values
    Object.keys(criteria).forEach(key => criteria[key] === undefined && delete criteria[key]);
    
    try {
        const response = await axios.post(`${API_BASE}/filter`, criteria);
        const results = response.data.data;
        
        const tbody = document.getElementById('filterTableBody');
        
        if (results.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 40px;">No stocks match your criteria</td></tr>';
            return;
        }
        
        tbody.innerHTML = results.map(stock => `
            <tr onclick="selectStockFromFilter('${stock.stock}')">
                <td><strong>${stock.stock}</strong></td>
                <td>₹${stock.price.toFixed(2)}</td>
                <td class="${stock.returns_1d >= 0 ? 'positive' : 'negative'}">${stock.returns_1d.toFixed(2)}%</td>
                <td class="${stock.returns_5d >= 0 ? 'positive' : 'negative'}">${stock.returns_5d.toFixed(2)}%</td>
                <td class="${stock.returns_1m >= 0 ? 'positive' : 'negative'}">${stock.returns_1m.toFixed(2)}%</td>
                <td>${stock.volume.toLocaleString()}</td>
            </tr>
        `).join('');
        
    } catch (error) {
        showError('Failed to filter stocks: ' + error.message);
    }
}

// Select stock from filter
function selectStockFromFilter(stockName) {
    document.getElementById('stockSelect').value = stockName;
    switchTab('analysis');
    loadStockData();
}

// Load strategies
async function loadStrategies() {
    try {
        const response = await axios.get(`${API_BASE}/strategies`);
        const strategies = response.data.strategies;
        
        // Display each strategy
        for (const [name, strategy] of Object.entries(strategies)) {
            const container = document.getElementById(`strategy${name.charAt(0).toUpperCase() + name.slice(1)}`);
            
            container.innerHTML = `
                <div class="metric">
                    <span class="metric-label">Buy Threshold</span>
                    <span class="metric-value">${strategy.buy_threshold.toFixed(2)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Sell Threshold</span>
                    <span class="metric-value">${strategy.sell_threshold.toFixed(2)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Min Confidence</span>
                    <span class="metric-value">${strategy.confidence_min}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Position Size</span>
                    <span class="metric-value">${(strategy.position_size * 100).toFixed(0)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Stop Loss</span>
                    <span class="metric-value">${strategy.stop_loss}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Take Profit</span>
                    <span class="metric-value">${strategy.take_profit}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Status</span>
                    <span class="metric-value">${strategy.enabled ? '✅ Enabled' : '❌ Disabled'}</span>
                </div>
            `;
        }
        
    } catch (error) {
        showError('Failed to load strategies: ' + error.message);
    }
}

// ============================================================================
// DATABASE CACHE FUNCTIONS - Fast loading from local database
// ============================================================================

async function syncDatabaseFromCSV() {
    const button = event.target;
    button.disabled = true;
    button.textContent = '⏳ Syncing...';
    
    try {
        const response = await axios.post(`${API_BASE}/db/sync-all`);
        
        if (response.data.success) {
            alert(`✅ Database Synced!\n\nSuccess: ${response.data.results.success} stocks\nFailed: ${response.data.results.failed} stocks\n\nYou can now use "Load from Database" for instant loading!`);
            button.textContent = '✅ Synced!';
        } else {
            alert('❌ Sync failed: ' + response.data.error);
            button.textContent = '🔄 Sync Database';
            button.disabled = false;
        }
    } catch (error) {
        alert('❌ Error syncing database: ' + error.message);
        button.textContent = '🔄 Sync Database';
        button.disabled = false;
    }
}

async function loadFromDatabase() {
    const button = event.target;
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '⏳ Loading...';
    
    document.getElementById('stocksLoading').style.display = 'inline';
    document.getElementById('loadProgress').textContent = 'Loading from database...';
    
    try {
        const response = await axios.get(`${API_BASE}/db/all-stocks-fast`);
        
        if (response.data.success) {
            /* Convert database format to our format */
            allStocksData = response.data.data.map(stock => ({
                stock: stock.stock_name,
                price: stock.today_close,
                change: stock.today_change,
                changePct: stock.today_pct,
                yesterdayPct: stock.yesterday_pct,
                day2Pct: stock.day2_pct || 0,
                day3Pct: stock.day3_pct || 0,
                day4Pct: stock.day4_pct || 0,
                day5Pct: stock.day5_pct || 0,
                day6Pct: stock.day6_pct || 0,
                week7DayPct: stock.week_7day_pct || 0,
                week7to14Pct: stock.week_7to14_pct || 0,
                week14to21Pct: stock.week_14to21_pct || 0,
                week21to31Pct: stock.week_21to31_pct || 0,
                volume: stock.volume,
                volumeRatio: stock.volume_ratio || 0,
                high52w: stock.week_52_high,
                low52w: stock.week_52_low,
                position52w: stock.week_52_position || 0,
                predictedPrice: stock.predicted_price || null,
                predictedChangePct: stock.predicted_change_pct || null,
                confidence: stock.confidence || null
            }));
            
            /* Hide loading */
            document.getElementById('stocksLoading').style.display = 'none';
            let message = `✅ Loaded ${allStocksData.length} stocks from database (INSTANT!)`;
            if (response.data.has_predictions) {
                message += ' | Predictions available 🔮';
            }
            document.getElementById('stocksCount').textContent = message;
            
            /* Default sort by 7-day % */
            currentSortColumn = 9;
            currentSortDirection = 'desc';
            allStocksData.sort((a, b) => b.week7DayPct - a.week7DayPct);
            
            /* Display the list */
            displayAllStocks();
            
            button.textContent = originalText;
            button.disabled = false;
        } else {
            throw new Error(response.data.error || 'Failed to load from database');
        }
    } catch (error) {
        document.getElementById('stocksLoading').style.display = 'none';
        
        if (error.response && error.response.status === 500) {
            alert('⚠️ Database not initialized!\n\nClick "Sync Database First" to load all stocks into the database.\nThis is a one-time operation.');
        } else {
            alert('❌ Error: ' + error.message);
        }
        
        button.textContent = originalText;
        button.disabled = false;
    }
}

async function updateTodayOnly() {
    const button = event.target;
    button.disabled = true;
    button.textContent = '⏳ Downloading CSV & Syncing...';
    
    try {
        /* Warn user this takes time */
        if (!confirm('⏱️ This will:\n\n1. Re-download ALL CSV files (2-5 minutes)\n2. Sync pending data to database\n\nThis ensures you have the latest prices for all stocks.\n\nContinue?')) {
            button.textContent = '🔄 Update Pending Data';
            button.disabled = false;
            return;
        }
        
        const response = await axios.post(`${API_BASE}/db/update-today`);
        
        if (response.data.success) {
            alert(`✅ Database Synced with Pending Data!\n\n` +
                `Updated: ${response.data.results.success} stocks\n` +
                `New Records: ${response.data.results.new_records}\n` +
                `No New Data: ${response.data.results.no_new_data}\n` +
                `Failed: ${response.data.results.failed}\n\n` +
                `Now refresh the list to see updated prices!`);
            button.textContent = '✅ Updated!';
            setTimeout(() => {
                button.textContent = '🔄 Update Pending Data';
                button.disabled = false;
            }, 3000);
        } else {
            throw new Error(response.data.error);
        }
    } catch (error) {
        alert('❌ Error updating: ' + error.message);
        button.textContent = '🔄 Update Pending Data';
        button.disabled = false;
    }
}

async function showDatabaseStats() {
    try {
        const response = await axios.get(`${API_BASE}/db/stats`);
        
        if (response.data.success) {
            const stats = response.data.stats;
            alert(`📊 Database Statistics\n\n` +
                `Total Records: ${stats.total_records.toLocaleString()}\n` +
                `Total Stocks: ${stats.total_stocks}\n` +
                `Date Range: ${stats.date_range}\n` +
                `Database Size: ${stats.db_size_mb} MB\n` +
                `File: ${stats.db_file}`);
        }
    } catch (error) {
        alert('❌ Error: ' + error.message);
    }
}

async function generatePredictions() {
    const button = event.target;
    const originalText = button.textContent;
    
    const limit = prompt('Generate predictions for how many stocks?\n\nRecommended: 50-100 (takes 20-60 seconds)', '50');
    if (!limit) return;
    
    button.disabled = true;
    button.textContent = '⏳ Generating...';
    
    try {
        const response = await axios.post(`${API_BASE}/db/predict-all`, {
            limit: parseInt(limit),
            force_refresh: false
        });
        
        if (response.data.success) {
            const results = response.data.results;
            alert(`🔮 Predictions Generated!\n\n` +
                `Success: ${results.success} stocks\n` +
                `Skipped: ${results.skipped} (already exists)\n` +
                `Failed: ${results.failed}\n` +
                `Prediction Date: ${response.data.prediction_date}\n\n` +
                `Now refresh the list to see predictions!`);
            button.textContent = originalText;
            button.disabled = false;
        } else {
            throw new Error(response.data.error);
        }
    } catch (error) {
        alert('❌ Error generating predictions: ' + error.message);
        button.textContent = originalText;
        button.disabled = false;
    }
}

// ============================================================================
// ALL STOCKS LIST TAB - Shows all stocks with prices and sorting
// ============================================================================

// Load all stocks with current prices
let allStocksData = [];
let currentSortColumn = -1;
let currentSortDirection = 'desc';

async function loadAllStocks() {
    try {
        hideError();
        
        // Show loading
        document.getElementById('stocksLoading').style.display = 'inline';
        document.getElementById('allStocksTableBody').innerHTML = 
            '<tr><td colspan="8" style="text-align: center; padding: 40px;">Loading stocks...</td></tr>';
        
        // Get all stocks
        const response = await axios.get(`${API_BASE}/stocks`);
        const stocks = response.data.data;
        
        document.getElementById('loadProgress').textContent = `0/${stocks.length}`;
        
        // Load data for all stocks (limit to first 100 for performance)
        const stocksToLoad = stocks.slice(0, 100);
        allStocksData = [];
        
        let loaded = 0;
        for (const stock of stocksToLoad) {
            try {
                const stockResponse = await axios.get(`${API_BASE}/stock/${stock}`, {
                    params: { period: '1m' }
                });
                
                if (stockResponse.data.success && stockResponse.data.data.length > 0) {
                    const data = stockResponse.data.data;
                    const summary = stockResponse.data.summary;
                    
                    // Calculate yesterday's change
                    let yesterdayChange = 0;
                    if (data.length >= 2) {
                        const yesterday = data[data.length - 2].Close;
                        const dayBefore = data.length >= 3 ? data[data.length - 3].Close : yesterday;
                        yesterdayChange = dayBefore !== 0 ? ((yesterday - dayBefore) / dayBefore * 100) : 0;
                    }
                    
                    allStocksData.push({
                        stock: stock,
                        price: summary.current_price || 0,
                        change: summary.change || 0,
                        changePct: summary.change_pct || 0,
                        yesterdayPct: yesterdayChange,
                        volume: summary.volume || 0,
                        high52w: summary.high_52w || 0,
                        low52w: summary.low_52w || 0
                    });
                }
                
                loaded++;
                document.getElementById('loadProgress').textContent = `${loaded}/${stocksToLoad.length}`;
                
            } catch (error) {
                // Skip stocks that fail to load
                loaded++;
            }
        }
        
        // Hide loading
        document.getElementById('stocksLoading').style.display = 'none';
        document.getElementById('stocksCount').textContent = 
            `Showing ${allStocksData.length} stocks (Limited to top 100 for performance)`;
        
        // Display stocks
        displayAllStocks();
        
    } catch (error) {
        showError('Failed to load stocks: ' + error.message);
        document.getElementById('stocksLoading').style.display = 'none';
    }
}

function displayAllStocks() {
    const tbody = document.getElementById('allStocksTableBody');
    
    if (allStocksData.length === 0) {
        tbody.innerHTML = '<tr><td colspan="16" style="text-align: center; padding: 40px;">No data available</td></tr>';
        return;
    }
    
    tbody.innerHTML = allStocksData.map(stock => {
        /* Helper function to get class and symbol */
        const getClass = (val) => val > 0 ? 'price-up' : val < 0 ? 'price-down' : 'price-neutral';
        const getSymbol = (val) => val >= 0 ? '+' : '';
        
        /* Prediction display */
        const predictedPriceHtml = stock.predictedPrice ? 
            `₹${stock.predictedPrice.toFixed(2)}` : 
            '<span style="color: #999;">--</span>';
        
        const predictedPctHtml = stock.predictedChangePct !== null ? 
            `<span class="${getClass(stock.predictedChangePct)}">${getSymbol(stock.predictedChangePct)}${stock.predictedChangePct.toFixed(2)}%</span>` : 
            '<span style="color: #999;">--</span>';
        
        return `
            <tr onclick="selectStockFromList('${stock.stock}')" style="cursor: pointer;">
                <td><strong>${stock.stock}</strong></td>
                <td>₹${stock.price.toFixed(2)}</td>
                <td class="${getClass(stock.changePct)}">${getSymbol(stock.changePct)}${stock.changePct.toFixed(2)}%</td>
                <td class="${getClass(stock.yesterdayPct)}">${getSymbol(stock.yesterdayPct)}${stock.yesterdayPct.toFixed(2)}%</td>
                <td class="${getClass(stock.day2Pct)}">${getSymbol(stock.day2Pct)}${stock.day2Pct.toFixed(2)}%</td>
                <td class="${getClass(stock.day3Pct)}">${getSymbol(stock.day3Pct)}${stock.day3Pct.toFixed(2)}%</td>
                <td class="${getClass(stock.day4Pct)}">${getSymbol(stock.day4Pct)}${stock.day4Pct.toFixed(2)}%</td>
                <td class="${getClass(stock.day5Pct)}">${getSymbol(stock.day5Pct)}${stock.day5Pct.toFixed(2)}%</td>
                <td class="${getClass(stock.day6Pct)}">${getSymbol(stock.day6Pct)}${stock.day6Pct.toFixed(2)}%</td>
                <td class="${getClass(stock.week7DayPct)}"><strong>${getSymbol(stock.week7DayPct)}${stock.week7DayPct.toFixed(2)}%</strong></td>
                <td class="${getClass(stock.week7to14Pct)}">${getSymbol(stock.week7to14Pct)}${stock.week7to14Pct.toFixed(2)}%</td>
                <td class="${getClass(stock.week14to21Pct)}">${getSymbol(stock.week14to21Pct)}${stock.week14to21Pct.toFixed(2)}%</td>
                <td class="${getClass(stock.week21to31Pct)}">${getSymbol(stock.week21to31Pct)}${stock.week21to31Pct.toFixed(2)}%</td>
                <td>${predictedPriceHtml}</td>
                <td>${predictedPctHtml}</td>
                <td>${stock.volume.toLocaleString()}</td>
            </tr>
        `;
    }).join('');
}

function sortStocksTable(columnIndex) {
    // Determine sort direction
    if (currentSortColumn === columnIndex) {
        currentSortDirection = currentSortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        currentSortColumn = columnIndex;
        currentSortDirection = 'desc';
    }
    
    // Sort data
    allStocksData.sort((a, b) => {
        let aVal, bVal;
        
        switch(columnIndex) {
            case 0: // Stock name
                aVal = a.stock;
                bVal = b.stock;
                break;
            case 1: // Price
                aVal = a.price;
                bVal = b.price;
                break;
            case 2: // Today %
                aVal = a.changePct;
                bVal = b.changePct;
                break;
            case 3: // Yesterday %
                aVal = a.yesterdayPct;
                bVal = b.yesterdayPct;
                break;
            case 4: // Day-2 %
                aVal = a.day2Pct;
                bVal = b.day2Pct;
                break;
            case 5: // Day-3 %
                aVal = a.day3Pct;
                bVal = b.day3Pct;
                break;
            case 6: // Day-4 %
                aVal = a.day4Pct;
                bVal = b.day4Pct;
                break;
            case 7: // Day-5 %
                aVal = a.day5Pct;
                bVal = b.day5Pct;
                break;
            case 8: // Day-6 %
                aVal = a.day6Pct;
                bVal = b.day6Pct;
                break;
            case 9: // Last 7 days %
                aVal = a.week7DayPct;
                bVal = b.week7DayPct;
                break;
            case 10: // 7-14 days %
                aVal = a.week7to14Pct;
                bVal = b.week7to14Pct;
                break;
            case 11: // 14-21 days %
                aVal = a.week14to21Pct;
                bVal = b.week14to21Pct;
                break;
            case 12: // 21-31 days %
                aVal = a.week21to31Pct;
                bVal = b.week21to31Pct;
                break;
            case 13: // Predicted Price
                aVal = a.predictedPrice || 0;
                bVal = b.predictedPrice || 0;
                break;
            case 14: // Predicted %
                aVal = a.predictedChangePct || 0;
                bVal = b.predictedChangePct || 0;
                break;
            case 15: // Volume
                aVal = a.volume;
                bVal = b.volume;
                break;
            default:
                return 0;
        }
        
        if (typeof aVal === 'string') {
            return currentSortDirection === 'asc' 
                ? aVal.localeCompare(bVal)
                : bVal.localeCompare(aVal);
        } else {
            return currentSortDirection === 'asc' 
                ? aVal - bVal
                : bVal - aVal;
        }
    });
    
    // Redisplay
    displayAllStocks();
}

function selectStockFromList(stockName) {
    document.getElementById('stockSelect').value = stockName;
    switchTab('analysis');
    loadStockData();
}

// Switch tab
function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Find and activate the clicked tab
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach((tab, index) => {
        if (tab.textContent.toLowerCase().includes(tabName === 'list' ? 'all stocks' : tabName)) {
            tab.classList.add('active');
        }
    });
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${tabName}-tab`).classList.add('active');
}

// Show error
function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

// Hide error
function hideError() {
    document.getElementById('error').style.display = 'none';
}

