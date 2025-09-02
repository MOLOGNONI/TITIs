// =============================================================================
// INITIALIZATION & UI EFFECTS
// =============================================================================

// Create floating particles for background
function createParticles() {
    const particlesContainer = document.getElementById('particles');
    if (!particlesContainer) return;
    for (let i = 0; i < 30; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 15 + 's';
        particle.style.animationDuration = (Math.random() * 10 + 10) + 's';
        particlesContainer.appendChild(particle);
    }
}

// Glitch effect for errors
function addGlitchEffect(element) {
    element.classList.add('glitch');
    setTimeout(() => element.classList.remove('glitch'), 300);
}

// =============================================================================
// CORE CALCULATION LOGIC
// =============================================================================

// Global SNE state
let sne = null;
let sessionStart = Date.now();

// Main calculation function - entry point from HTML button
function calculateUncertainty() {
    // If SNE is active, delegate to the enhanced function
    if (sne) {
        calculateUncertaintyWithSNE();
        return;
    }

    const loading = document.getElementById('loading');
    loading.style.display = 'block';

    // Get input values
    const r = parseFloat(document.getElementById('r_value').value) || 0;
    const R = parseFloat(document.getElementById('R_value').value) || 0;
    const sw = parseFloat(document.getElementById('sw_value').value) || 0;
    const BL = parseFloat(document.getElementById('BL_value').value) || 0;
    const sa = parseFloat(document.getElementById('sa_value').value) || 0;
    const sb = parseFloat(document.getElementById('sb_value').value) || 0;
    const n = parseInt(document.getElementById('n_value').value) || 10;

    // Validate inputs
    if (r <= 0 || R <= 0) {
        alert('⚠️ ERRO: Valores de repetibilidade e reprodutibilidade devem ser positivos!');
        addGlitchEffect(document.querySelector('.panel'));
        loading.style.display = 'none';
        return;
    }

    // Simulate processing delay
    setTimeout(() => {
        try {
            // Calculations based on ISO 21748
            const sr = r / 2.8;
            const sR = R / 2.8;
            const sL = Math.sqrt(Math.max(0, Math.pow(sR, 2) - Math.pow(sr, 2))); // Prevent negative sqrt
            const sD = Math.sqrt(Math.pow(sL, 2) + (Math.pow(sw, 2) / n));
            const two_sD = 2 * sD;
            const u_prime = sL;
            const u = Math.sqrt(Math.pow(u_prime, 2) + Math.pow(sa, 2) + Math.pow(sb, 2));
            const U = 2 * u;

            const biasValid = Math.abs(BL) <= two_sD;
            const precisionValid = sw < 1.5 * sr;

            updateResults({ sr, sR, sL, sD, u_prime, u, U, bias_limit: two_sD });
            updateValidationStatus(biasValid, precisionValid, BL, two_sD, sw, sr);

        } catch (error) {
            console.error('Calculation error:', error);
            alert('⚠️ ERRO DE CÁLCULO: Verifique os dados inseridos!');
            addGlitchEffect(document.querySelector('.panel'));
        }
        loading.style.display = 'none';
    }, 1500);
}

// Enhanced calculation function with SNE integration
function calculateUncertaintyWithSNE() {
    const loading = document.getElementById('loading');
    loading.style.display = 'block';

    const inputData = {
        r: parseFloat(document.getElementById('r_value').value) || 0,
        R: parseFloat(document.getElementById('R_value').value) || 0,
        sw: parseFloat(document.getElementById('sw_value').value) || 0,
        BL: parseFloat(document.getElementById('BL_value').value) || 0,
        sa: parseFloat(document.getElementById('sa_value').value) || 0,
        sb: parseFloat(document.getElementById('sb_value').value) || 0,
        n: parseInt(document.getElementById('n_value').value) || 10
    };

    if (sne) {
        sne.detectUniversalPatterns(inputData);
        sne.updateMemoryOperator('calculation', inputData);
    }

    if (inputData.r <= 0 || inputData.R <= 0) {
        alert('⚠️ ERRO: Valores de repetibilidade e reprodutibilidade devem ser positivos!');
        addGlitchEffect(document.querySelector('.panel'));
        loading.style.display = 'none';
        return;
    }

    setTimeout(() => {
        try {
            const sr = inputData.r / 2.8;
            const sR = inputData.R / 2.8;
            const sL = Math.sqrt(Math.max(0, Math.pow(sR, 2) - Math.pow(sr, 2)));
            const sD = Math.sqrt(Math.pow(sL, 2) + (Math.pow(inputData.sw, 2) / inputData.n));
            const two_sD = 2 * sD;
            const u_prime = sL;
            const u = Math.sqrt(Math.pow(u_prime, 2) + Math.pow(inputData.sa, 2) + Math.pow(inputData.sb, 2));
            const U = 2 * u;

            const biasValid = Math.abs(inputData.BL) <= two_sD;
            const precisionValid = inputData.sw < 1.5 * sr;

            const results = { sr, sR, sL, sD, u_prime, u, U, bias_limit: two_sD };

            updateResults(results);
            updateValidationStatus(biasValid, precisionValid, inputData.BL, two_sD, inputData.sw, sr);

            if (sne) {
                sne.updateMemoryOperator('results', results);
            }
        } catch (error) {
            console.error('SNE Calculation error:', error);
            alert('⚠️ ERRO DE CÁLCULO SNE: Verifique os dados inseridos!');
            addGlitchEffect(document.querySelector('.panel'));
        }
        loading.style.display = 'none';
    }, 1500);
}


// =============================================================================
// UI & DATA HANDLING FUNCTIONS
// =============================================================================

function updateResults(results) {
    document.getElementById('sr_result').textContent = results.sr.toFixed(4);
    document.getElementById('sR_result').textContent = results.sR.toFixed(4);
    document.getElementById('sL_result').textContent = results.sL.toFixed(4);
    document.getElementById('sD_result').textContent = results.sD.toFixed(4);
    document.getElementById('u_prime_result').textContent = results.u_prime.toFixed(4);
    document.getElementById('u_result').textContent = results.u.toFixed(4);
    document.getElementById('U_result').textContent = results.U.toFixed(4);
    document.getElementById('bias_limit_result').textContent = results.bias_limit.toFixed(4);

    document.querySelectorAll('.result-value').forEach(el => {
        el.style.animation = 'none';
        setTimeout(() => { el.style.animation = 'glow 1s ease-in-out'; }, 10);
    });
}

function updateValidationStatus(biasValid, precisionValid, BL, two_sD, sw, sr) {
    const biasStatusText = document.getElementById('biasStatusText');
    const biasIndicator = document.getElementById('biasIndicator');
    const precisionStatusText = document.getElementById('precisionStatusText');
    const precisionIndicator = document.getElementById('precisionIndicator');

    if (biasValid) {
        biasStatusText.textContent = `APROVADO (|${BL.toFixed(3)}| ≤ ${two_sD.toFixed(3)})`;
        biasStatusText.style.color = '#00ff00';
        biasIndicator.className = 'status-indicator status-ok';
    } else {
        biasStatusText.textContent = `REPROVADO (|${BL.toFixed(3)}| > ${two_sD.toFixed(3)})`;
        biasStatusText.style.color = '#ff0080';
        biasIndicator.className = 'status-indicator status-error';
    }

    if (precisionValid) {
        precisionStatusText.textContent = `APROVADO (${sw.toFixed(3)} < ${(1.5 * sr).toFixed(3)})`;
        precisionStatusText.style.color = '#00ff00';
        precisionIndicator.className = 'status-indicator status-ok';
    } else {
        precisionStatusText.textContent = `ATENÇÃO (${sw.toFixed(3)} ≥ ${(1.5 * sr).toFixed(3)})`;
        precisionStatusText.style.color = '#ffff00';
        precisionIndicator.className = 'status-indicator status-warning';
    }
}

function clearInputs() {
    document.querySelectorAll('.cyber-input').forEach(input => {
        if (input.id !== 'n_value') {
            input.value = '';
        }
    });
    document.querySelectorAll('.result-value').forEach(el => {
        el.textContent = '-';
    });
    document.getElementById('biasStatusText').textContent = 'Não calculado';
    document.getElementById('precisionStatusText').textContent = 'Não calculado';
    document.getElementById('biasIndicator').className = 'status-indicator';
    document.getElementById('precisionIndicator').className = 'status-indicator';
}

function extractCurrentResults() {
    const sr = document.getElementById('sr_result').textContent;
    if (sr === '-') return null;
    return {
        'sr': sr,
        'sR': document.getElementById('sR_result').textContent,
        'sL': document.getElementById('sL_result').textContent,
        'sD': document.getElementById('sD_result').textContent,
        'u_prime': document.getElementById('u_prime_result').textContent,
        'u': document.getElementById('u_result').textContent,
        'U': document.getElementById('U_result').textContent,
        'bias_limit': document.getElementById('bias_limit_result').textContent
    };
}

function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// =============================================================================
// EXPORT FUNCTIONS
// =============================================================================

function exportToCSV() {
    const results = extractCurrentResults();
    if (!results) { alert('⚠️ Calcule os resultados primeiro!'); return; }
    let csv = 'Parâmetro,Valor\n';
    Object.entries(results).forEach(([key, value]) => { csv += `${key},${value}\n`; });
    downloadFile(csv, 'iso21748_results.csv', 'text/csv');
}

function exportToJSON() {
    const results = extractCurrentResults();
    if (!results) { alert('⚠️ Calcule os resultados primeiro!'); return; }
    const data = {
        timestamp: new Date().toISOString(),
        calculation_method: 'ISO 21748:2017',
        results: results,
        validation: {
            bias_status: document.getElementById('biasStatusText').textContent,
            precision_status: document.getElementById('precisionStatusText').textContent
        },
        sne_enabled: sne !== null
    };
    const json = JSON.stringify(data, null, 2);
    downloadFile(json, 'iso21748_results.json', 'application/json');
}

function generateReport() {
    const results = extractCurrentResults();
    if (!results) { alert('⚠️ Calcule os resultados primeiro!'); return; }
    const sneSection = sne ? `
SNE FRAMEWORK DATA:
- Tempo de sessão: ${Math.round((Date.now() - sessionStart) / 1000)} segundos
- Padrões detectados: ${sne.usagePatterns.length}
- Evolução temporal: ${sne.temporalEvolution.toFixed(2)}
- Memória operador: ${sne.memoryOperator.size} registros
            ` : '';
    const report = `
ISO 21748 - RELATÓRIO DE INCERTEZA DE MEDIÇÃO
==============================================
Data: ${new Date().toLocaleString('pt-BR')}
Método: ISO 21748:2017 - Cyberpunk Enhanced
DADOS DE ENTRADA:
- Repetibilidade (r): ${document.getElementById('r_value').value}
- Reprodutibilidade (R): ${document.getElementById('R_value').value}
- Desvio Padrão Intralaboratorial (sw): ${document.getElementById('sw_value').value}
- Bias Laboratorial (BL): ${document.getElementById('BL_value').value}
- Incertezas Adicionais (sa, sb): ${document.getElementById('sa_value').value}, ${document.getElementById('sb_value').value}
- Número de Replicatas (n): ${document.getElementById('n_value').value}
RESULTADOS CALCULADOS:
- sr (Desvio Repetibilidade): ${results.sr}
- sR (Desvio Reprodutibilidade): ${results.sR}
- sL (Desvio Interlaboratorial): ${results.sL}
- sD (Desvio Experimental): ${results.sD}
- u' (Incerteza Provisória): ${results.u_prime}
- u (Incerteza Final): ${results.u}
- U (Incerteza Expandida): ${results.U}
VALIDAÇÃO:
- Critério Bias: ${document.getElementById('biasStatusText').textContent}
- Critério Precisão: ${document.getElementById('precisionStatusText').textContent}
${sneSection}
CONCLUSÃO:
Incerteza Expandida (U) = ${results.U} (k=2, 95% confiança)
Gerado por: ISO 21748 Cyber Lab Analyzer v2.0 SNE-Enhanced
Framework: MUCA XAI / SNE Project
            `;
    downloadFile(report, 'iso21748_report.txt', 'text/plain');
}

function exportSNEData() {
    if (!sne) { alert('⚠️ SNE Framework não está ativo!'); return; }
    const sneData = {
        timestamp: new Date().toISOString(),
        usage_patterns: sne.usagePatterns,
        temporal_evolution: sne.temporalEvolution,
        memory_operator_size: sne.memoryOperator.size,
        session_duration: Date.now() - sessionStart,
        pattern_insights: sne.lucaDetector.generateInsights(sne.usagePatterns)
    };
    const json = JSON.stringify(sneData, null, 2);
    downloadFile(json, 'sne_data_export.json', 'application/json');
}

// =============================================================================
// SNE FRAMEWORK INTEGRATION
// =============================================================================

function toggleSNEMode() {
    const sneStatus = document.getElementById('sneStatus');
    const patternIndicator = document.getElementById('patternIndicator');
    if (!sne) {
        sne = new SNEEnhancedInterface();
        sneStatus.style.display = 'block';
        patternIndicator.classList.add('active');
        document.querySelectorAll('.panel').forEach(panel => panel.classList.add('temporal-glow'));
        updateSNEInfo('Sistema inicializado - Detectando padrões...');
    } else {
        sne = null;
        sneStatus.style.display = 'none';
        patternIndicator.classList.remove('active');
        document.querySelectorAll('.panel').forEach(panel => panel.classList.remove('temporal-glow'));
    }
}

function updateSNEInfo(message) {
    const sneInfo = document.getElementById('sneInfo');
    if (sneInfo) { sneInfo.textContent = message; }
}

class SNEEnhancedInterface {
    constructor() {
        this.usagePatterns = JSON.parse(localStorage.getItem('sne_patterns') || '[]');
        this.temporalEvolution = 0;
        this.memoryOperator = new Map();
        this.lucaDetector = new PatternDetector();
        this.initializeTemporalOperator();
    }
    initializeTemporalOperator() { setInterval(() => { this.temporalEvolution += 0.1; this.adaptInterface(); }, 5000); }
    detectUniversalPatterns(inputData) {
        const pattern = { timestamp: Date.now(), values: inputData, complexity: this.calculateComplexity(inputData), entropy: this.calculateEntropy(inputData) };
        this.usagePatterns.push(pattern);
        this.savePatterns();
        return this.lucaDetector.findUniversalPattern(this.usagePatterns);
    }
    updateMemoryOperator(action, result) {
        const memoryKey = `${action}_${Date.now()}`;
        this.memoryOperator.set(memoryKey, { action, result, context: this.getCurrentContext(), success: this.evaluateSuccess(result) });
        if (this.memoryOperator.size > 100) { this.memoryOperator.delete(this.memoryOperator.keys().next().value); }
    }
    adaptInterface() {
        const recentPatterns = this.getRecentPatterns();
        if (recentPatterns.length > 5) {
            const avgComplexity = recentPatterns.reduce((sum, p) => sum + p.complexity, 0) / recentPatterns.length;
            if (avgComplexity > 0.7) { this.enhancePrecision(); }
            if (recentPatterns.length > 10) { this.optimizePerformance(); }
            this.displayPatternInsights(recentPatterns);
        }
    }
    calculateComplexity(data) {
        const values = Object.values(data).filter(v => typeof v === 'number');
        if (values.length < 2) return 0;
        const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        return Math.min(1, variance / (mean + 1));
    }
    calculateEntropy(data) {
        const values = Object.values(data).filter(v => typeof v === 'number');
        const uniqueValues = [...new Set(values)];
        return uniqueValues.length / (values.length + 1);
    }
    getCurrentContext() { return { timeOfDay: new Date().getHours(), dayOfWeek: new Date().getDay(), temporalEvolution: this.temporalEvolution, sessionLength: Date.now() - (sessionStart || Date.now()) }; }
    evaluateSuccess(result) { const resultValues = Object.values(result).filter(v => v !== '-' && v !== null); return resultValues.length / 8; }
    getRecentPatterns() { const tenMinutesAgo = Date.now() - (10 * 60 * 1000); return this.usagePatterns.filter(p => p.timestamp > tenMinutesAgo); }
    enhancePrecision() {
        document.querySelectorAll('.result-value').forEach(el => {
            if (el.textContent !== '-') { const value = parseFloat(el.textContent); if (!isNaN(value)) { el.textContent = value.toFixed(6); } }
        });
        this.showTemporaryMessage('🧬 PRECISÃO AUMENTADA - Padrões complexos detectados');
    }
    optimizePerformance() { document.documentElement.style.setProperty('--animation-speed', '0.5s'); this.showTemporaryMessage('⚡ PERFORMANCE OTIMIZADA - Uso frequente detectado'); }
    displayPatternInsights(patterns) { const insights = this.lucaDetector.generateInsights(patterns); if (insights && insights.length > 0) { this.showPatternInsight(insights[0]); } }
    showTemporaryMessage(message) {
        const msgDiv = document.createElement('div');
        msgDiv.textContent = message;
        msgDiv.style.cssText = `position: fixed; top: 20px; right: 20px; z-index: 1000; background: rgba(0, 255, 255, 0.9); color: #000; padding: 10px 20px; border-radius: 5px; font-weight: bold; font-size: 12px; animation: fadeInOut 3s ease-in-out forwards;`;
        document.body.appendChild(msgDiv);
        setTimeout(() => msgDiv.remove(), 3000);
    }
    showPatternInsight(insight) {
        const insightDiv = document.createElement('div');
        insightDiv.innerHTML = `<div style="color: #ffff00; font-weight: bold;">🧠 INSIGHT LUCA DETECTADO:</div><div style="color: #00ffff; font-size: 11px; margin-top: 5px;">${insight}</div>`;
        insightDiv.style.cssText = `position: fixed; bottom: 20px; left: 20px; z-index: 1000; background: rgba(0, 0, 0, 0.9); padding: 15px; border-radius: 8px; border: 1px solid #ffff00; max-width: 300px; animation: slideInLeft 0.5s ease-out;`;
        document.body.appendChild(insightDiv);
        setTimeout(() => insightDiv.remove(), 8000);
    }
    savePatterns() { if (this.usagePatterns.length > 50) { this.usagePatterns = this.usagePatterns.slice(-50); } localStorage.setItem('sne_patterns', JSON.stringify(this.usagePatterns)); }
}

class PatternDetector {
    findUniversalPattern(patterns) {
        if (patterns.length < 3) return null;
        const recentPatterns = patterns.slice(-10);
        const complexityTrend = this.analyzeTrend(recentPatterns.map(p => p.complexity));
        const entropyTrend = this.analyzeTrend(recentPatterns.map(p => p.entropy));
        return { complexityTrend, entropyTrend, patternStability: this.calculateStability(recentPatterns), universalSignature: this.extractUniversalSignature(patterns) };
    }
    analyzeTrend(values) {
        if (values.length < 2) return 'insufficient_data';
        const slope = this.calculateSlope(values);
        if (Math.abs(slope) < 0.01) return 'stable';
        return slope > 0 ? 'increasing' : 'decreasing';
    }
    calculateSlope(values) {
        const n = values.length; const sumX = (n * (n - 1)) / 2; const sumY = values.reduce((s, v) => s + v, 0); const sumXY = values.reduce((s, v, i) => s + i * v, 0); const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6;
        return (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    }
    calculateStability(patterns) {
        if (patterns.length < 2) return 1;
        const complexities = patterns.map(p => p.complexity); const mean = complexities.reduce((s, v) => s + v, 0) / complexities.length; const variance = complexities.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / complexities.length;
        return Math.max(0, 1 - variance);
    }
    extractUniversalSignature(patterns) {
        const signatures = patterns.map(p => ({ complexity_bucket: Math.floor(p.complexity * 10), entropy_bucket: Math.floor(p.entropy * 10), time_of_day: new Date(p.timestamp).getHours() }));
        const signatureMap = new Map();
        signatures.forEach(sig => { const key = `${sig.complexity_bucket}_${sig.entropy_bucket}_${Math.floor(sig.time_of_day / 4)}`; signatureMap.set(key, (signatureMap.get(key) || 0) + 1); });
        const mostCommon = [...signatureMap.entries()].sort((a, b) => b[1] - a[1])[0];
        return mostCommon ? mostCommon[0] : 'unique_pattern';
    }
    generateInsights(patterns) {
        const pattern = this.findUniversalPattern(patterns);
        if (!pattern) return [];
        const insights = [];
        if (pattern.complexityTrend === 'increasing') insights.push('Dados estão ficando mais complexos - possível evolução do método ou mudança de matriz');
        else if (pattern.complexityTrend === 'decreasing') insights.push('Padronização detectada - processo laboratorial estabilizando');
        if (pattern.patternStability > 0.8) insights.push('Alta estabilidade nos padrões - método bem controlado');
        else if (pattern.patternStability < 0.3) insights.push('Variabilidade alta detectada - revisar procedimentos analíticos');
        if (pattern.entropyTrend === 'increasing') insights.push('Aumento na diversidade de dados - expandindo escopo analítico');
        return insights;
    }
}

function createMemoryTrace(element) {
    const trace = document.createElement('div');
    trace.className = 'memory-trace';
    const rect = element.getBoundingClientRect();
    trace.style.left = rect.left + 'px';
    trace.style.top = rect.top + 'px';
    trace.style.setProperty('--dx', Math.random() * 200 - 100 + 'px');
    trace.style.setProperty('--dy', Math.random() * 200 - 100 + 'px');
    document.body.appendChild(trace);
    setTimeout(() => trace.remove(), 5000);
}

// =============================================================================
// EVENT LISTENERS & STARTUP
// =============================================================================

document.addEventListener('keydown', function(event) {
    if (event.ctrlKey && event.key === 'Enter') { document.querySelector('.cyber-button[onclick="calculateUncertainty()"]').click(); }
    if (event.ctrlKey && event.key === 'd') { event.preventDefault(); clearInputs(); }
    if (event.ctrlKey && event.key === 's') { event.preventDefault(); toggleSNEMode(); }
});

document.querySelectorAll('.cyber-input').forEach(input => {
    input.addEventListener('input', function() {
        if (this.type === 'number') { this.value = this.value.replace(/[^0-9.-]/g, ''); }
        if (sne) { createMemoryTrace(this); }
    });
    input.addEventListener('focus', function() { this.style.boxShadow = '0 0 20px rgba(0, 255, 255, 0.8)'; });
    input.addEventListener('blur', function() { this.style.boxShadow = ''; });
});

window.addEventListener('load', function() {
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 1s ease-in-out';
    setTimeout(() => { document.body.style.opacity = '1'; }, 100);

    createParticles();

    setTimeout(() => {
        if (confirm('🧬 Ativar SNE Framework para demonstração? (Detecta padrões e evolui a interface)')) {
            toggleSNEMode();
        }
    }, 2000);
});
